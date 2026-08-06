import xarray as xr
import xclim.indices as xci
import numpy as np
import pandas as pd
from open_climate_service.process import process

@process(
    summary="calculate SPI-1 index", # type: ignore
    parameters={
        "baseline": {
            "description": "baseline period for SPI calculation (e.g., 1990-2020)" # type: ignore
        } # type: ignore
    },
)
def spi_1(pr: xr.DataArray, baseline: str) -> xr.DataArray:
    """
    Calculate the Standardized Precipitation Index (SPI) for the specified baseline period.

    Parameters:
    - pr (xr.DataArray): Input precipitation data array.
    - baseline (str): The baseline period for SPI calculation (e.g., "1990-2020").
    """
    try:
        start_year, end_year = baseline.split("-")
        cal_start = f"{start_year.strip()}-01-01"
        cal_end = f"{end_year.strip()}-12-31"
    except ValueError:
        raise ValueError("Baseline must be in the format 'YYYY-YYYY', e.g., '1990-2020'")

    if not np.issubdtype(pr.time.dtype, np.datetime64):
        pr = pr.assign_coords(time=pd.to_datetime(pr.time.values))
    pr = pr.sortby("time")

    pr_monthly = pr.resample(time="MS").sum(dim="time")

    pr_monthly.attrs["units"] = "mm/day"

    spi_1 = xci.standardized_precipitation_index(  # type: ignore
        pr=pr_monthly,
        freq="MS",
        window=1,
        dist="gamma",
        method="ML",
        cal_start=cal_start,
        cal_end=cal_end,
    )
    
    spi_1.name = "spi_1"
    spi_1.attrs["long_name"] = "Standardized Precipitation Index (SPI) - 1 month"
    spi_1.attrs["description"] = (
        "The Standardized Precipitation Index (SPI) is a widely used index to characterize "
        "meteorological drought. It is calculated based on the probability of precipitation "
        "for a given time scale, in this case, 1 month. The SPI-1 index indicates the deviation "
        "of precipitation from the long-term average for the specified baseline period."
    )

    return spi_1