"""Processes for estimating population exposed to gridded hazards."""
from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  # activates the .rio accessor
import xarray as xr
from open_climate_service.process import process
from rasterio.enums import Resampling

TIME_DIMS = ("time", "t")
BAND_DIMS = ("band", "bands")
SPATIAL_DIMS = ("y", "x")
SPATIAL_DIM_SET = set(SPATIAL_DIMS)
SPATIAL_ALIASES = {
    "latitude": "y",
    "lat": "y",
    "longitude": "x",
    "lon": "x",
}
VALID_SEVERITIES = (0, 1, 2, 3, 4)
OUTPUT_VARIABLES = ("exposed_population", "hazard_severity")


def _snapshot(da: xr.DataArray, name: str) -> tuple[xr.DataArray, np.ndarray | None]:
    """Reduce to a 2-D (y, x) raster, returning the time value if there was one."""
    time_value = None
    for dim in TIME_DIMS:
        if dim in da.dims:
            if da.sizes[dim] != 1:
                raise ValueError(f"{name} must contain exactly one time step")
            if dim in da.coords:
                time_value = da[dim].values
            da = da.isel({dim: 0}, drop=True)
            break

    for dim in BAND_DIMS:
        if dim in da.dims:
            if da.sizes[dim] != 1:
                raise ValueError(f"{name} must contain exactly one band")
            da = da.isel({dim: 0}, drop=True)
            break

    # Normalize coordinate names (e.g. lat/lon -> y/x)
    for alias, standard in SPATIAL_ALIASES.items():
        if alias in da.dims:
            da = da.rename({alias: standard})

    extra = set(da.dims) - SPATIAL_DIM_SET
    if extra or set(da.dims) != SPATIAL_DIM_SET:
        raise ValueError(
            f"{name} must reduce to dimensions (y, x); got {tuple(da.dims)}"
        )

    # Ensure canonical (y, x) dimension ordering
    da = da.transpose("y", "x")

    if da.rio.crs is None:
        raise ValueError(f"{name} raster has no CRS")
    return da, time_value


def _as_float_with_nan(da: xr.DataArray) -> xr.DataArray:
    """Cast to float and turn any declared nodata value into NaN."""
    nodata = da.rio.nodata
    crs = da.rio.crs
    da = da.astype("float32")
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata)
    da = da.rio.write_nodata(np.nan)
    if crs is not None:
        da = da.rio.write_crs(crs)
    return da


@process(
    summary="Overlay hazard severity with WorldPop on the population grid",
    parameters={
        "hazard": {
            "description": (
                "A single-snapshot hazard raster with categories 0-4; "
                "0 means no hazard. Other values are treated as missing."
            )
        },
        "population": {
            "description": "A single-year raster of population counts per cell."
        },
        "variable": {
            "description": (
                "Variable to return: 'exposed_population' or 'hazard_severity' "
                "(one variable per published dataset), or 'all' for both."
            )
        },
    },
)
def population_exposure_by_hazard(
    hazard: xr.DataArray, population: xr.DataArray, variable: str = "all"
) -> xr.Dataset:
    """Return exposed population and hazard severity on the WorldPop grid.

    Severity is transferred to the population grid with nearest-neighbour
    resampling. Population in cells with severity 1-4 is retained; cells with
    severity 0 have an exposed-population value of zero. Hazard values outside
    0-4, hazard nodata, cells outside the hazard extent, and cells with missing
    population are missing in ``exposed_population``; hazard cells that are
    missing or invalid stay missing in ``hazard_severity``.
    """
    if variable not in ("all", *OUTPUT_VARIABLES):
        raise ValueError(
            f"variable must be 'all' or one of {OUTPUT_VARIABLES}; got {variable!r}"
        )

    hazard_snapshot, hazard_time = _snapshot(hazard, "hazard")
    population_snapshot, pop_time = _snapshot(population, "population")

    target_crs = population_snapshot.rio.crs

    # Nodata and out-of-range handling happen before reprojection so that
    # padding from reproject_match is NaN, never a fake "0 = no hazard".
    hazard_snapshot = _as_float_with_nan(hazard_snapshot)
    hazard_snapshot = hazard_snapshot.where(hazard_snapshot.isin(VALID_SEVERITIES))
    population_snapshot = _as_float_with_nan(population_snapshot)

    severity = hazard_snapshot.rio.reproject_match(
        population_snapshot,
        resampling=Resampling.nearest,
        nodata=np.nan,
    )
    # Guard against float jitter in coordinates from the reprojection.
    if severity.shape != population_snapshot.shape:
        raise ValueError("reprojected hazard does not match the population grid")
    severity = severity.assign_coords(
        x=population_snapshot["x"], y=population_snapshot["y"]
    )
    population_snapshot, severity = xr.align(
        population_snapshot, severity, join="exact", copy=False
    )

    valid = severity.notnull() & population_snapshot.notnull()

    # BUG FIX: Use severity >= 1 so Category 1 is retained as exposed
    exposed = (
        xr.where(severity >= 1, population_snapshot, 0.0)
        .where(valid)
        .rename("exposed_population")
    )
    severity = severity.rename("hazard_severity")

    # BUG FIX: Re-write CRS to prevent dropping grid mapping info after xr.where
    exposed = exposed.rio.write_crs(target_crs)
    severity = severity.rio.write_crs(target_crs)

    # Fresh metadata
    exposed.attrs = {
        "long_name": "Population exposed to the hazard",
        "units": "people",
    }
    severity.attrs = {
        "long_name": "Hazard severity category",
        "units": "1",
        "flag_values": list(VALID_SEVERITIES),
        "flag_meanings": "none category_1 category_2 category_3 category_4",
    }
    exposed.encoding = {}
    severity.encoding = {}

    # Prefer hazard time snapshot, fallback to population year if hazard is static
    out_time = hazard_time if hazard_time is not None else pop_time
    if out_time is not None:
        exposed = exposed.expand_dims(time=out_time)
        severity = severity.expand_dims(time=out_time)

    result = xr.Dataset(
        {
            "exposed_population": exposed,
            "hazard_severity": severity,
        }
    )
    result = result.rio.write_crs(target_crs)

    return result if variable == "all" else result[[variable]]