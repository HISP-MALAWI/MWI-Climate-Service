import numpy as np
import xarray as xr
from open_climate_service.process import process


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

    tmin, tmax = xr.align(tmin, tmax, join="exact", copy=False)
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
