from open_climate_service.process import process
import xarray as xr
import xclim.indices as xci

@process(
    summary="calculate SPI indice", # type: ignore
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
    - baseline (str): The baseline period for SPI calculation (e.g., "1990-2020").

    Returns:
    - xr.Dataset: A dataset containing the calculated SPI values.
    """
    
    try:
        start_year, end_year = baseline.split("-")
    except ValueError:
        raise ValueError("Baseline must be in the format 'YYYY-YYYY', e.g., '1990-2020'")

    pr_cal = pr.sel(time=slice(start_year, end_year))

    spi_1 = xci.standardized_precipitation_index( # type: ignore
        pr=pr,
        pr_cal=pr_cal,
        freq="MS",
        window=1,
        dist="gamma",
        method="MLE",
    )

    spi_1.name = "spi-1"
    spi_1.attrs["long_name"] = "Standardized Precipitation Index (SPI) - 1 month"
    spi_1.attrs["description"] = (
        "The Standardized Precipitation Index (SPI) is a widely used index to characterize "
        "meteorological drought. It is calculated based on the probability of precipitation "
        "for a given time scale, in this case, 1 month. The SPI-1 index indicates the deviation "
        "of precipitation from the long-term average for the specified baseline period."
    )

    return spi_1