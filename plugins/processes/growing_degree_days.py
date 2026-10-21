import numpy as np
import pandas as pd
import xarray as xr
from open_climate_service.process import process

TIME_DIMS = ("t", "time", "valid_time", "date", "day")


def _align_daily_temperatures(
    tmin: xr.DataArray, tmax: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Normalize daily time labels before requiring exact coordinate alignment."""
    tmin_time = next((dim for dim in TIME_DIMS if dim in tmin.dims), None)
    tmax_time = next((dim for dim in TIME_DIMS if dim in tmax.dims), None)
    if (tmin_time is None) != (tmax_time is None):
        raise ValueError("tmin and tmax must both have a time dimension or neither")
    if tmin_time is None:
        return xr.align(tmin, tmax, join="exact", copy=False)

    if tmin_time != tmax_time:
        tmax = tmax.rename({tmax_time: tmin_time})
        tmax_time = tmin_time
    if tmin_time not in tmin.coords or tmax_time not in tmax.coords:
        raise ValueError("tmin and tmax time dimensions must have coordinates")

    try:
        tmin_days = pd.DatetimeIndex(tmin[tmin_time].values).normalize()
        tmax_days = pd.DatetimeIndex(tmax[tmax_time].values).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("tmin and tmax time coordinates must be datetime-like") from exc

    if tmin_days.has_duplicates or tmax_days.has_duplicates:
        raise ValueError("tmin and tmax must have at most one value per day")
    if not tmin_days.equals(tmax_days):
        tmin_dates = set(tmin_days.strftime("%Y-%m-%d"))
        tmax_dates = set(tmax_days.strftime("%Y-%m-%d"))
        only_tmin = sorted(tmin_dates - tmax_dates)
        only_tmax = sorted(tmax_dates - tmin_dates)
        raise ValueError(
            "tmin and tmax must cover the same dates; "
            f"only in tmin: {only_tmin[:5]}, only in tmax: {only_tmax[:5]}"
        )

    tmin = tmin.assign_coords({tmin_time: tmin_days.to_numpy()})
    tmax = tmax.assign_coords({tmax_time: tmax_days.to_numpy()})
    return xr.align(tmin, tmax, join="exact", copy=False)


@process(
    summary="Calculate daily Growing Degree Days",
    parameters={
        "tmin": {
            "description": "Daily minimum temperature in degrees Celsius."
        },
        "tmax": {
            "description": "Daily maximum temperature in degrees Celsius."
        },
        "base_temperature": {
            "description": (
                "Crop-specific base temperature in degrees Celsius. "
                "Use 10 for the common 10°C base."
            )
        },
    },
)
def growing_degree_days(
    tmin: xr.DataArray,
    tmax: xr.DataArray,
    base_temperature: float,
) -> xr.DataArray:
    """Calculate daily GDD as max(0, ((Tmin + Tmax) / 2) - Tbase)."""
    if not np.isfinite(base_temperature):
        raise ValueError("base_temperature must be a finite number in degrees Celsius")

    tmin, tmax = _align_daily_temperatures(tmin, tmax)
    gdd = (((tmin + tmax) / 2) - base_temperature).clip(min=0)
    gdd.name = "gdd"
    gdd.attrs.update(
        units="degree_Celsius day",
        long_name="Daily Growing Degree Days",
        description=(
            "Daily heat accumulation calculated as the mean of daily minimum "
            "and maximum temperature minus the base temperature, floored at zero."
        ),
    )
    return gdd
