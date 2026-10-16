"""Processes for estimating population exposed to gridded categorical hazards.

Works for any classified hazard raster (heat stress, flood depth, drought
class, cyclone wind band, ...). The class codes and labels come, in order of
priority, from: the ``hazard_categories`` / ``no_hazard_value`` parameters,
the classes declared on the hazard cube (CF ``flag_values`` / ``flag_meanings``,
stamped from the dataset template by ``load_collection``), and finally the
legacy default of codes 1-4.
"""
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

# Fallback only: used when neither the caller nor the dataset declares classes.
DEFAULT_HAZARD_CATEGORIES = (1, 2, 3, 4)
NO_HAZARD_VALUE = 0

CLASS_VALUES_ATTR = "flag_values"
CLASS_MEANINGS_ATTR = "flag_meanings"

OUTPUT_VARIABLES = (
    "exposed_population",
    "hazard_severity",
    "population_by_hazard_category",
)
POPULATION_BY_HAZARD_CATEGORY_VARIABLE = "population_by_hazard_category"
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


def _parse_categories(category, valid_categories) -> tuple[int, ...] | None:
    """None/'all' -> None; otherwise parse and validate selected categories."""
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
            f"category must be 'all' or values from {valid_categories}; "
            f"got {category!r}"
        ) from exc
    bad = [v for v in values if v not in valid_categories]
    if bad or not values:
        raise ValueError(
            f"category must be 'all' or values from {valid_categories}; "
            f"got {category!r}"
        )
    return tuple(sorted(set(values)))


def _parse_hazard_categories(
    categories, no_hazard_value: int | None
) -> tuple[int, ...]:
    """Parse the configured integer codes that represent exposed hazard classes."""
    try:
        if isinstance(categories, str):
            values = [
                int(part)
                for part in categories.replace(";", ",").split(",")
                if part.strip()
            ]
        else:
            values = [int(value) for value in categories]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hazard_categories must be a non-empty list of integer category codes"
        ) from exc

    parsed = tuple(sorted(set(values)))
    if not parsed or (no_hazard_value is not None and no_hazard_value in parsed):
        raise ValueError(
            "hazard_categories must be non-empty and must not include "
            f"no_hazard_value ({no_hazard_value})"
        )
    return parsed


# ------------------------------------------------------- declared classes
def _declared_classes(hazard: xr.DataArray) -> dict[int, str]:
    """Return {code: label} from CF flag attributes on the hazard, or {}.

    ``flag_meanings`` is a space-separated string in CF, so underscores stand
    in for spaces in a label; they are turned back into spaces here.
    """
    values = hazard.attrs.get(CLASS_VALUES_ATTR)
    if values is None:
        return {}
    codes = [int(v) for v in np.atleast_1d(values)]
    meanings = hazard.attrs.get(CLASS_MEANINGS_ATTR)
    if isinstance(meanings, str):
        labels = meanings.split()
    elif meanings is not None:
        labels = [str(m) for m in meanings]
    else:
        labels = []
    if len(labels) != len(codes):
        labels = [str(c) for c in codes]
    return {c: l.replace("_", " ") for c, l in zip(codes, labels)}


def _resolve_hazard_categories(
    categories, declared: dict[int, str], no_hazard_value: int | None
) -> tuple[int, ...]:
    """Explicit parameter > classes declared on the data > legacy default."""
    if categories is not None:
        return _parse_hazard_categories(categories, no_hazard_value)
    exposed = tuple(sorted(c for c in declared if c != no_hazard_value))
    return exposed or DEFAULT_HAZARD_CATEGORIES


def _flag_meanings(codes, declared: dict[int, str]) -> str:
    return " ".join(declared.get(c, str(c)).replace(" ", "_") for c in codes)


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
    """Split the hazard into (day, 2-D raster) pairs, one per time step."""
    hazard = _promote_scalar_time(hazard)
    time_dim = _find_time_dim(hazard)

    # Multi-band hazard (one band per day): bands become the days.
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


def _severity_on_population_grid(hazard_2d, grid, valid_values) -> xr.DataArray:
    # Mask before reprojection so padding and any nodata code (for example 255)
    # stay NaN instead of being read as a hazard class.
    hazard_2d = _as_float_with_nan(hazard_2d)
    hazard_2d = hazard_2d.where(hazard_2d.isin(valid_values))
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
    summary="Population exposed to a categorical hazard (any hazard x population)",
    parameters={
        "hazard": {
            "description": (
                "Categorical hazard raster for any hazard type (heat stress, "
                "flood, drought, ...). Class codes and labels are read from "
                "the classes declared on the dataset (CF flag_values / "
                "flag_meanings) unless hazard_categories and no_hazard_value "
                "are given. Other values, including the nodata code, are "
                "treated as missing. Steps may be a time / valid_time / date "
                "/ day / t dimension, a scalar time coordinate, or bands "
                "(one per step)."
            )
        },
        "population": {
            "description": (
                "Yearly WorldPop counts per cell: one raster, or a stack with "
                "a time (year) dimension. Resampled to each hazard step."
            )
        },
        "variable": {
            "description": (
                "'population_by_hazard_category' (default): population in "
                "each exposed class. 'exposed_population': total over the "
                "selected classes. 'hazard_severity': the class grid. "
                "'all': exposed_population and hazard_severity."
            )
        },
        "category": {
            "description": (
                "Optional class filter, chosen from the exposed classes. "
                "Accepts one code, comma-separated codes, or 'all' (default). "
                "A single class gives one band per step."
            )
        },
        "hazard_categories": {
            "description": (
                "Integer codes that count as exposed. Defaults to the "
                "declared classes of the hazard dataset except "
                "no_hazard_value, or 1, 2, 3, 4 if none are declared."
            )
        },
        "no_hazard_value": {
            "description": (
                "Integer code meaning no hazard (default: 0). It is a valid "
                "value but never counts as exposed. Use null if the hazard "
                "has no such class."
            )
        },
        "date": {
            "description": (
                "Optional day (YYYY-MM-DD); only hazard steps on that day are "
                "used. Omit to process every step."
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
    variable: str = POPULATION_BY_HAZARD_CATEGORY_VARIABLE,
    category: str | int | None = None,
    date: str | None = None,
    population_resampling: str = "step",
    hazard_start_date: str | None = None,
    hazard_categories: tuple[int, ...] | list[int] | str | None = None,
    no_hazard_value: int | None = NO_HAZARD_VALUE,
) -> xr.Dataset:
    """Exposed population on the WorldPop grid, split by hazard class.

    For every hazard step the hazard is reprojected (nearest neighbour) onto
    the population grid and combined with that step's population, taken from
    the yearly WorldPop rasters (see ``population_resampling``). Output time
    always follows the hazard steps; the population's own year never labels it.

    ``population_by_hazard_category`` bins population by each exposed class and
    has dims (time, exposure_level, y, x), dropping ``exposure_level`` when one
    class is selected. The ``exposure_level`` coordinate carries
    ``flag_values`` / ``flag_meanings`` and an ``exposure_level_label``
    coordinate with the class labels. ``exposed_population`` sums the selected
    classes (all exposed classes by default). Hazard values that are neither an
    exposed class nor ``no_hazard_value``, cells outside the hazard extent, and
    missing or negative population are NaN. Sum in float64 for totals.
    """
    if variable not in ("all", *OUTPUT_VARIABLES):
        raise ValueError(
            f"variable must be 'all' or one of {OUTPUT_VARIABLES}; "
            f"got {variable!r}"
        )
    if population_resampling not in POPULATION_METHODS:
        raise ValueError(
            f"population_resampling must be one of {POPULATION_METHODS}; "
            f"got {population_resampling!r}"
        )

    declared = _declared_classes(hazard)  # read before the cube is split by step
    hazard_levels = _resolve_hazard_categories(
        hazard_categories, declared, no_hazard_value
    )
    categories = _parse_categories(category, hazard_levels)

    levels = categories or hazard_levels

    steps = _hazard_steps(hazard, date, hazard_start_date)
    rasters = _load_population(population)
    grid = next(iter(rasters.values()))
    target_crs = grid.rio.crs
    valid_hazard_values = tuple(
        v for v in (no_hazard_value, *hazard_levels) if v is not None
    )

    want_levels = variable == POPULATION_BY_HAZARD_CATEGORY_VARIABLE
    want_exposed = variable in ("all", "exposed_population")
    want_severity = variable in ("all", "hazard_severity")

    sev_steps, exp_steps, lvl_steps = [], [], []
    for day, hazard_2d in steps:
        severity = _severity_on_population_grid(
            hazard_2d, grid, valid_hazard_values
        )
        pop_day = _population_for_day(rasters, day, population_resampling)
        valid = severity.notnull() & pop_day.notnull()

        outputs = {}
        if want_severity:
            outputs["sev"] = severity
        if want_exposed:
            mask = severity.isin(categories or hazard_levels)
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

    hazard_name = hazard.attrs.get("long_name") or hazard.name or "hazard"

    data = {}
    if want_levels:
        cube = _stack(lvl_steps)
        cube = cube.transpose(
            *[d for d in ("time", "exposure_level", "y", "x") if d in cube.dims]
        )
        cube = cube.assign_coords(
            exposure_level_label=(
                "exposure_level",
                [declared.get(lv, str(lv)) for lv in levels],
            )
        )
        if len(levels) == 1:  # one class -> one band per step
            cube = cube.isel(exposure_level=0)
        cube = cube.rename(POPULATION_BY_HAZARD_CATEGORY_VARIABLE)
        cube.attrs = {
            "long_name": f"Population exposed at each {hazard_name} class",
            "units": "people",
        }
        cube.coords["exposure_level"].attrs = {
            "long_name": f"{hazard_name} exposure class",
            "units": "1",
            "flag_values": list(levels),
            "flag_meanings": _flag_meanings(levels, declared),
        }
        cube.encoding = {}
        data[POPULATION_BY_HAZARD_CATEGORY_VARIABLE] = cube
    if want_exposed:
        exposed = _stack(exp_steps).rename("exposed_population")
        exposed.attrs = {
            "long_name": f"Population exposed to {hazard_name}"
            + (f" (classes {list(categories)})" if categories else ""),
            "units": "people",
        }
        exposed.encoding = {}
        data["exposed_population"] = exposed
    if want_severity:
        sev = _stack(sev_steps).rename("hazard_severity")
        codes = sorted(valid_hazard_values)
        sev.attrs = {
            "long_name": f"{hazard_name} severity class",
            "units": "1",
            "flag_values": codes,
            "flag_meanings": _flag_meanings(codes, declared),
        }
        sev.encoding = {}
        data["hazard_severity"] = sev

    result = xr.Dataset(data).rio.write_crs(target_crs)
    result.attrs["population_resampling"] = population_resampling
    return result