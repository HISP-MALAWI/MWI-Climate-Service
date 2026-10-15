"""Processes for estimating population exposed to gridded hazards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  # activates the .rio accessor
import xarray as xr
from open_climate_service.process import process
from rasterio.enums import Resampling

# Names a time axis may use. "valid_time" is first because in forecast data
# "time" is often the (scalar) forecast reference time.
TIME_NAMES = ("valid_time", "time", "date", "day", "t")
BAND_DIMS = ("band", "bands")
SPATIAL_DIMS = ("y", "x")
SPATIAL_DIM_SET = set(SPATIAL_DIMS)
SPATIAL_ALIASES = {"latitude": "y", "lat": "y", "longitude": "x", "lon": "x"}
VALID_SEVERITIES = (0, 1, 2, 3, 4)
OUTPUT_VARIABLES = ("exposed_population", "hazard_severity")
HEATSTRESS_EXPOSURE_VARIABLE = "heatstress_exposure"
HEATSTRESS_EXPOSURE_LEVELS = (1, 2, 3)
POPULATION_METHODS = ("step", "linear")


# ---------------------------------------------------------------- helpers
def _find_time_dim(da: xr.DataArray) -> str | None:
    return next((name for name in TIME_NAMES if name in da.dims), None)


def _promote_scalar_time(da: xr.DataArray) -> xr.DataArray:
    """Turn a scalar time coordinate into a length-1 dimension."""
    if _find_time_dim(da) is None:
        for name in TIME_NAMES:
            if name in da.coords and da[name].ndim == 0:
                return da.expand_dims(name)
    return da


def _snapshot(da: xr.DataArray, name: str) -> xr.DataArray:
    """Reduce to a 2-D (y, x) raster. Time is handled by the caller."""
    for dim in (*TIME_NAMES, *BAND_DIMS):
        if dim in da.dims:
            if da.sizes[dim] != 1:
                raise ValueError(f"{name} must contain exactly one '{dim}' step")
            da = da.isel({dim: 0}, drop=True)

    for alias, standard in SPATIAL_ALIASES.items():
        if alias in da.dims:
            da = da.rename({alias: standard})

    if set(da.dims) != SPATIAL_DIM_SET:
        raise ValueError(
            f"{name} must reduce to dimensions (y, x); got {tuple(da.dims)}"
        )
    da = da.transpose("y", "x")

    # Drop leftover scalar coords (time, step, number, ...) but keep the CRS.
    stray = [
        c for c in da.coords
        if c not in da.dims and c != "spatial_ref" and da[c].ndim == 0
    ]
    da = da.drop_vars(stray)

    if da.rio.crs is None:
        raise ValueError(f"{name} raster has no CRS")
    return da


def _as_float_with_nan(da: xr.DataArray) -> xr.DataArray:
    """Cast to float and turn any declared nodata value into NaN."""
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata)  # mask in the original dtype
    return da.astype("float32").rio.write_nodata(np.nan)


def _parse_categories(category) -> tuple[int, ...] | None:
    """None/'all' -> None; 2, '2', '1,3' or [1, 3] -> sorted tuple."""
    if category is None:
        return None
    try:
        if isinstance(category, str):
            text = category.strip().lower()
            if text in ("", "all"):
                return None
            values = [int(p) for p in text.replace(";", ",").split(",") if p.strip()]
        elif isinstance(category, (int, np.integer)):
            values = [int(category)]
        else:
            values = [int(v) for v in category]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"category must be 'all' or values from {HEATSTRESS_EXPOSURE_LEVELS}; "
            f"got {category!r}"
        ) from exc
    bad = [v for v in values if v not in HEATSTRESS_EXPOSURE_LEVELS]
    if bad or not values:
        raise ValueError(
            f"category must be 'all' or values from {HEATSTRESS_EXPOSURE_LEVELS}; "
            f"got {category!r}"
        )
    return tuple(sorted(set(values)))


# ------------------------------------------------------------- hazard days
def _select_indices(times: np.ndarray, date) -> list[int]:
    if date is None:
        return list(range(len(times)))
    target = pd.Timestamp(date).normalize()
    try:
        days = pd.DatetimeIndex(times).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hazard time coordinate is not datetime-like; cannot select a date"
        ) from exc
    idx = np.flatnonzero(days == target).tolist()
    if not idx:
        raise ValueError(
            f"hazard has no time step on {target.date()}; "
            f"available {days.min().date()} to {days.max().date()}"
        )
    return idx


def _hazard_steps(hazard, date, start_date) -> list[tuple[object, xr.DataArray]]:
    """Split the hazard into (day, 2-D raster) pairs, one per forecast day."""
    hazard = _promote_scalar_time(hazard)
    time_dim = _find_time_dim(hazard)

    # Multi-band hazard (one band per forecast day): bands become the days.
    if time_dim is None:
        band_dim = next(
            (b for b in BAND_DIMS if b in hazard.dims and hazard.sizes[b] > 1), None
        )
        if band_dim is not None:
            n = hazard.sizes[band_dim]
            band_values = hazard[band_dim].values if band_dim in hazard.coords else None
            if band_values is not None and np.issubdtype(band_values.dtype, np.datetime64):
                times = band_values
            elif start_date is not None:
                times = pd.date_range(start_date, periods=n, freq="D").values
            else:
                raise ValueError(
                    f"hazard has {n} bands but no dates; pass hazard_start_date "
                    "(day of the first band) or give it a time dimension"
                )
            hazard = hazard.rename({band_dim: "time"}).assign_coords(
                time=("time", times)
            )
            time_dim = "time"

    if time_dim is None:
        if date is not None:
            raise ValueError("date was given but the hazard has no time information")
        return [(None, _snapshot(hazard, "hazard"))]

    if time_dim not in hazard.coords:
        raise ValueError(f"hazard time dimension '{time_dim}' must have a coordinate")

    times = hazard[time_dim].values
    return [
        (times[i], _snapshot(hazard.isel({time_dim: i}), "hazard"))
        for i in _select_indices(times, date)
    ]


def _severity_on_population_grid(hazard_2d, grid) -> xr.DataArray:
    # Mask before reprojection so padding is NaN, never "0 = no hazard".
    hazard_2d = _as_float_with_nan(hazard_2d)
    hazard_2d = hazard_2d.where(hazard_2d.isin(VALID_SEVERITIES))
    severity = hazard_2d.rio.reproject_match(
        grid, resampling=Resampling.nearest, nodata=np.nan
    )
    if severity.shape != grid.shape:
        raise ValueError("reprojected hazard does not match the population grid")
    if not (
        np.allclose(severity["x"], grid["x"]) and np.allclose(severity["y"], grid["y"])
    ):
        raise ValueError("reprojected hazard coordinates drift from the population grid")
    return severity.assign_coords(x=grid["x"], y=grid["y"])


# ------------------------------------------------------- yearly population
def _to_years(values: np.ndarray) -> list[int]:
    if np.issubdtype(values.dtype, np.datetime64):
        return [int(y) for y in pd.DatetimeIndex(values).year]
    return [int(v) for v in values]


def _load_population(population: xr.DataArray) -> dict:
    """Return {year: 2-D float raster}; key None means 'no time axis'."""

    def prep(da, name="population"):
        da = _as_float_with_nan(_snapshot(da, name))
        return da.where(da >= 0)  # negative counts are undeclared nodata

    population = _promote_scalar_time(population)
    time_dim = _find_time_dim(population)
    if time_dim is None or (
        time_dim not in population.coords and population.sizes[time_dim] == 1
    ):
        return {None: prep(population)}
    if time_dim not in population.coords:
        raise ValueError(f"population time dimension '{time_dim}' needs a coordinate")

    years = _to_years(population[time_dim].values)
    if len(set(years)) != len(years):
        raise ValueError("population has duplicate years")
    rasters = {y: prep(population.isel({time_dim: i})) for i, y in enumerate(years)}

    first = next(iter(rasters.values()))
    for y, r in rasters.items():
        if (
            r.shape != first.shape
            or not np.allclose(r["x"], first["x"])
            or not np.allclose(r["y"], first["y"])
        ):
            raise ValueError(f"population year {y} is on a different grid")
    return rasters


def _population_for_day(rasters: dict, day, method: str) -> xr.DataArray:
    """Resample the yearly population to one day.

    step   : latest available year <= the day's year (earliest if before all).
    linear : interpolate between yearly rasters by fractional year; clamps to
             the first/last year outside the available range.
    """
    years = sorted(k for k in rasters if k is not None)
    if not years:
        return rasters[None]
    if day is None:
        return rasters[years[-1]]

    ts = pd.Timestamp(day)
    if method == "step":
        eligible = [y for y in years if y <= ts.year]
        return rasters[eligible[-1] if eligible else years[0]]

    frac = ts.year + (ts.dayofyear - 1) / (366 if ts.is_leap_year else 365)
    if frac <= years[0]:
        return rasters[years[0]]
    if frac >= years[-1]:
        return rasters[years[-1]]
    y0 = max(y for y in years if y <= frac)
    y1 = min(y for y in years if y > frac)
    w = (frac - y0) / (y1 - y0)
    return rasters[y0] * (1.0 - w) + rasters[y1] * w


def _stack(items: list[xr.DataArray]) -> xr.DataArray:
    if len(items) == 1:
        return items[0]
    return xr.concat(
        items, dim="time", join="exact", coords="minimal", compat="override"
    )


# ----------------------------------------------------------------- process
@process(
    summary="Daily population exposed to heat-stress categories (WorldPop x hazard)",
    parameters={
        "hazard": {
            "description": (
                "Daily hazard forecast with categories 0-4 (0 = no hazard; "
                "other values are treated as missing). Days may be a time / "
                "valid_time / date / day / t dimension, a scalar time "
                "coordinate, or bands (one per day)."
            )
        },
        "population": {
            "description": (
                "Yearly WorldPop counts per cell: one raster, or a stack with "
                "a time (year) dimension. Resampled to each hazard day."
            )
        },
        "variable": {
            "description": (
                "'heatstress_exposure' (default): exposed population by "
                "category, one band per day (and per category if several are "
                "kept). 'exposed_population': one total over the selected "
                "categories. 'hazard_severity': the daily category grid. "
                "'all': exposed_population and hazard_severity."
            )
        },
        "category": {
            "description": (
                "Filter by heat-stress category: 1, 2 or 3, several as '1,3', "
                "or 'all' (default). A single category gives one band per day."
            )
        },
        "date": {
            "description": (
                "Optional day (YYYY-MM-DD); only hazard steps on that day are "
                "used. Omit to process every forecast day."
            )
        },
        "population_resampling": {
            "description": (
                "'step' (default): use each day's year of WorldPop (latest "
                "available year if later). 'linear': interpolate between "
                "yearly rasters by day of year."
            )
        },
        "hazard_start_date": {
            "description": (
                "Only for a multi-band hazard without dates: the day (YYYY-MM-DD) "
                "of the first band; following bands are consecutive days."
            )
        },
    },
)
def population_exposure_by_hazard(
    hazard: xr.DataArray,
    population: xr.DataArray,
    variable: str = HEATSTRESS_EXPOSURE_VARIABLE,
    category: str | int | None = None,
    date: str | None = None,
    population_resampling: str = "step",
    hazard_start_date: str | None = None,
) -> xr.Dataset:
    """Daily exposed population on the WorldPop grid, split by category.

    For every hazard day the hazard is reprojected (nearest neighbour) onto the
    population grid and combined with that day's population, taken from the
    yearly WorldPop rasters (see ``population_resampling``). Output time always
    follows the hazard days; the population's own year never labels it.

    ``heatstress_exposure`` bins are exclusive (severity == category) for
    categories 1-3 and have dims (time, exposure_level, y, x); selecting a
    single category drops ``exposure_level`` leaving (time, y, x), i.e. one
    band per day. ``exposed_population`` counts cells whose severity is in the
    selected categories (severity 1-4 when no category is given). Hazard values
    outside 0-4, cells outside the hazard extent, and missing or negative
    population are NaN. Sum in float64 for totals.
    """
    if variable not in ("all", *OUTPUT_VARIABLES, HEATSTRESS_EXPOSURE_VARIABLE):
        raise ValueError(
            "variable must be 'all', one of "
            f"{OUTPUT_VARIABLES}, or {HEATSTRESS_EXPOSURE_VARIABLE!r}; "
            f"got {variable!r}"
        )
    if population_resampling not in POPULATION_METHODS:
        raise ValueError(
            f"population_resampling must be one of {POPULATION_METHODS}; "
            f"got {population_resampling!r}"
        )
    categories = _parse_categories(category)
    levels = categories or HEATSTRESS_EXPOSURE_LEVELS

    steps = _hazard_steps(hazard, date, hazard_start_date)
    rasters = _load_population(population)
    grid = next(iter(rasters.values()))
    target_crs = grid.rio.crs

    want_levels = variable == HEATSTRESS_EXPOSURE_VARIABLE
    want_exposed = variable in ("all", "exposed_population")
    want_severity = variable in ("all", "hazard_severity")

    sev_steps, exp_steps, lvl_steps = [], [], []
    for day, hazard_2d in steps:
        severity = _severity_on_population_grid(hazard_2d, grid)
        pop_day = _population_for_day(rasters, day, population_resampling)
        valid = severity.notnull() & pop_day.notnull()

        outputs = {}
        if want_severity:
            outputs["sev"] = severity
        if want_exposed:
            mask = severity.isin(categories) if categories else severity >= 1
            outputs["exp"] = xr.where(mask, pop_day, 0.0).where(valid)
        if want_levels:
            outputs["lvl"] = xr.concat(
                [xr.where(severity == lv, pop_day, 0.0).where(valid) for lv in levels],
                dim=xr.IndexVariable("exposure_level", list(levels)),
            )

        for key, arr in outputs.items():
            if day is not None:
                arr = arr.expand_dims(time=[day])
            {"sev": sev_steps, "exp": exp_steps, "lvl": lvl_steps}[key].append(arr)

    data = {}
    if want_levels:
        cube = _stack(lvl_steps)
        cube = cube.transpose(
            *[d for d in ("time", "exposure_level", "y", "x") if d in cube.dims]
        )
        if len(levels) == 1:  # one category -> one band per day
            cube = cube.isel(exposure_level=0)
        cube = cube.rename(HEATSTRESS_EXPOSURE_VARIABLE)
        cube.attrs = {
            "long_name": "Population exposed at each heat-stress category",
            "units": "people",
        }
        cube.coords["exposure_level"].attrs = {
            "long_name": "Heat-stress exposure category",
            "units": "1",
            "flag_values": list(levels),
        }
        cube.encoding = {}
        data[HEATSTRESS_EXPOSURE_VARIABLE] = cube
    if want_exposed:
        exposed = _stack(exp_steps).rename("exposed_population")
        exposed.attrs = {
            "long_name": "Population exposed to the hazard"
            + (f" (categories {list(categories)})" if categories else ""),
            "units": "people",
        }
        exposed.encoding = {}
        data["exposed_population"] = exposed
    if want_severity:
        sev = _stack(sev_steps).rename("hazard_severity")
        sev.attrs = {
            "long_name": "Hazard severity category",
            "units": "1",
            "flag_values": list(VALID_SEVERITIES),
            "flag_meanings": "none category_1 category_2 category_3 category_4",
        }
        sev.encoding = {}
        data["hazard_severity"] = sev

    result = xr.Dataset(data).rio.write_crs(target_crs)
    result.attrs["population_resampling"] = population_resampling
    return result