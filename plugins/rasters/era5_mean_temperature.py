"""Daily mean 2 m air temperature from the ERA5 reanalysis via CDS."""

from __future__ import annotations

import calendar
import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

_DATASET = "reanalysis-era5-single-levels"
_API_VARIABLE = "2m_temperature"
_FIRST_DATE = date(1940, 1, 1)
_PUBLICATION_LAG_DAYS = 5
_HOURLY_TIMES = [f"{hour:02d}:00" for hour in range(24)]


class Era5MeanTemperaturePlugin(BaseDatasetPlugin):
    """Ingestion plugin for daily means of ERA5 hourly 2 m temperature.

    ERA5 values are converted from kelvin to degrees Celsius after averaging
    the 24 hourly values for each UTC day. A calendar month is downloaded per
    CDS request and kept in memory for subsequent daily fetches.
    """

    max_concurrency = 1
    commit_batch_size = 31
    rechunk_time = 31
    pyramid: bool = True

    def __init__(self) -> None:
        self._cached_month: tuple[int, int] | None = None
        self._cached_ds: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        start_date = max(date.fromisoformat(start[:10]), _FIRST_DATE)
        end_date = min(
            date.fromisoformat(end[:10]),
            date.today() - timedelta(days=_PUBLICATION_LAG_DAYS),
        )
        if start_date > end_date:
            return []

        periods = []
        current = start_date
        while current <= end_date:
            periods.append(current.isoformat())
            current += timedelta(days=1)
        return periods

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        requested_date = date.fromisoformat(period_id[:10])
        month = (requested_date.year, requested_date.month)
        if self._cached_month != month:
            self._cached_ds = self._download_month(*month, bbox)
            self._cached_month = month

        if self._cached_ds is None:
            raise RuntimeError(f"ERA5 data cache was not populated for {month[0]}-{month[1]:02d}")

        time_coord = "valid_time" if "valid_time" in self._cached_ds.coords else "time"
        target = np.datetime64(requested_date.isoformat())
        try:
            da = self._cached_ds[_API_VARIABLE].sel({time_coord: target})
        except KeyError as exc:
            raise ValueError(f"ERA5 has no hourly temperature values for {period_id}") from exc

        rename = {
            name: target_name
            for name, target_name in (("latitude", "y"), ("longitude", "x"))
            if name in da.dims
        }
        if rename:
            da = da.rename(rename)
        da = (da - 273.15).astype("float32")
        da.attrs["units"] = "degC"
        da.attrs["standard_name"] = "air_temperature"
        da.attrs["cell_methods"] = "time: mean"

        result = da.to_dataset(name="tmean")
        result = result.expand_dims(t=[np.datetime64(requested_date.isoformat())])
        return result.load()

    def _download_month(self, year: int, month: int, bbox: list[float]) -> xr.Dataset:
        import cdsapi

        xmin, ymin, xmax, ymax = map(float, bbox)
        _, last_day = calendar.monthrange(year, month)
        cutoff = date.today() - timedelta(days=_PUBLICATION_LAG_DAYS)
        if (year, month) == (cutoff.year, cutoff.month):
            last_day = min(last_day, cutoff.day)
        logger.info("Downloading ERA5 2 m temperature for %d-%02d from CDS", year, month)

        request = {
            "product_type": ["reanalysis"],
            "variable": [_API_VARIABLE],
            "year": [f"{year:04d}"],
            "month": [f"{month:02d}"],
            "day": [f"{day:02d}" for day in range(1, last_day + 1)],
            "time": _HOURLY_TIMES,
            "area": [ymax, xmin, ymin, xmax],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "era5.nc"
            cdsapi.Client().retrieve(_DATASET, request, str(nc_path))
            with xr.open_dataset(nc_path) as downloaded:
                ds = downloaded.load()

        if "expver" in ds.dims:
            merged = ds.isel(expver=0, drop=True)
            for index in range(1, ds.sizes["expver"]):
                merged = merged.combine_first(ds.isel(expver=index, drop=True))
            ds = merged
        elif "expver" in ds.coords:
            ds = ds.drop_vars("expver")

        # ERA5's hourly surface field is in kelvin. Compute the daily UTC mean
        # before converting units, preserving the average for all 24 hours.
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        variable = (
            _API_VARIABLE
            if _API_VARIABLE in ds.data_vars
            else "t2m"
            if "t2m" in ds.data_vars
            else None
        )
        if variable is None:
            raise ValueError(
                "CDS ERA5 response does not contain the expected 2 m temperature "
                f"variable (found: {list(ds.data_vars)})"
            )
        if variable != _API_VARIABLE:
            ds = ds.rename({variable: _API_VARIABLE})
        ds = ds.resample({time_coord: "1D"}).mean()
        if time_coord != "time":
            ds = ds.rename({time_coord: "time"})
        return ds
