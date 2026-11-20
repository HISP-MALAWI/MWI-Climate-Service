"""Daily mean 2 m air temperature from ERA5-Land on DestinE Earth Data Hub."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

import aiohttp
import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

_ZARR_URL = "https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr"
_VARIABLE_CANDIDATES = ("2m_temperature", "t2m", "temperature_2m")
_FIRST_DATE = date(1950, 1, 1)
_PUBLICATION_LAG_DAYS = 5


class Era5MeanTemperaturePlugin(BaseDatasetPlugin):
    """Ingest daily means of hourly ERA5-Land 2 m temperature.

    Earth Data Hub serves the hourly source as a Zarr v3 store. A requested
    month is subset to the configured bbox, averaged to daily values, and
    cached while the ingestion processes each day's output period.
    """

    max_concurrency = 1
    commit_batch_size = 31
    rechunk_time = 31
    pyramid: bool = True

    def __init__(self) -> None:
        self._source: xr.Dataset | None = None
        self._cached_month: tuple[int, int] | None = None
        self._cached_daily: xr.Dataset | None = None

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
            self._cached_daily = self._read_month(*month, bbox)
            self._cached_month = month

        if self._cached_daily is None:
            raise RuntimeError(
                f"ERA5-Land cache was not populated for {month[0]}-{month[1]:02d}"
            )

        target = np.datetime64(requested_date.isoformat())
        try:
            daily = self._cached_daily["tmean"].sel(time=target)
        except KeyError as exc:
            raise ValueError(
                f"Earth Data Hub ERA5-Land has no 2 m temperature values for {period_id}"
            ) from exc

        result = daily.expand_dims(t=[target]).to_dataset(name="tmean")
        result["tmean"].attrs.update(
            {
                "units": "degC",
                "standard_name": "air_temperature",
                "cell_methods": "time: mean",
                "source": "ERA5-Land hourly 2 m temperature, averaged over UTC day",
            }
        )
        return result.load()

    def _open_source(self) -> xr.Dataset:
        if self._source is not None:
            return self._source

        api_key = os.getenv("EDH_API_KEY") or os.getenv("EDH_PASSWORD")
        if not api_key:
            raise ValueError(
                "DestinE Earth Data Hub API key missing. Set EDH_API_KEY in the "
                "OCS server environment (or EDH_PASSWORD for compatibility)."
            )

        username = os.getenv("EDH_USERNAME", "edh")
        auth = aiohttp.BasicAuth(username, api_key)
        logger.info("Opening ERA5-Land Zarr source from DestinE Earth Data Hub")
        self._source = xr.open_zarr(
            _ZARR_URL,
            storage_options={
                "client_kwargs": {
                    "auth": auth,
                    "trust_env": True,
                }
            },
            zarr_format=3,
        )
        return self._source

    def _read_month(self, year: int, month: int, bbox: list[float]) -> xr.Dataset:
        source = self._open_source()
        variable = next(
            (name for name in _VARIABLE_CANDIDATES if name in source.data_vars),
            None,
        )
        if variable is None:
            raise ValueError(
                "Earth Data Hub ERA5-Land store does not contain a recognized "
                f"2 m temperature variable; found {list(source.data_vars)}."
            )

        time_name = next(
            (name for name in ("valid_time", "time") if name in source[variable].coords),
            None,
        )
        latitude_name = next(
            (name for name in ("latitude", "lat", "y") if name in source[variable].coords),
            None,
        )
        longitude_name = next(
            (name for name in ("longitude", "lon", "x") if name in source[variable].coords),
            None,
        )
        if time_name is None or latitude_name is None or longitude_name is None:
            raise ValueError(
                "Earth Data Hub ERA5-Land temperature is missing time/latitude/"
                f"longitude coordinates: {source[variable].dims}."
            )

        xmin, ymin, xmax, ymax = map(float, bbox)
        first_day = date(year, month, 1)
        next_month = date(year + int(month == 12), month % 12 + 1, 1)
        time_values = source[time_name]
        selected = source[variable].where(
            (time_values >= np.datetime64(first_day.isoformat()))
            & (time_values < np.datetime64(next_month.isoformat())),
            drop=True,
        )
        selected = selected.where(
            (selected[longitude_name] >= xmin)
            & (selected[longitude_name] <= xmax)
            & (selected[latitude_name] >= ymin)
            & (selected[latitude_name] <= ymax),
            drop=True,
        )
        if selected.sizes.get(time_name, 0) == 0:
            raise ValueError(
                f"Earth Data Hub ERA5-Land has no hourly temperature data for "
                f"{year}-{month:02d}."
            )
        if (
            selected.sizes.get(latitude_name, 0) == 0
            or selected.sizes.get(longitude_name, 0) == 0
        ):
            raise ValueError(f"ERA5-Land has no source cells inside bbox {bbox}.")

        daily = selected.resample({time_name: "1D"}).mean(skipna=True)
        rename = {
            source_name: target_name
            for source_name, target_name in (
                (time_name, "time"),
                (latitude_name, "y"),
                (longitude_name, "x"),
            )
            if source_name != target_name
        }
        if rename:
            daily = daily.rename(rename)

        daily = (daily - 273.15).astype("float32")
        daily.name = "tmean"
        daily.attrs["units"] = "degC"
        daily.attrs["standard_name"] = "air_temperature"
        daily.attrs["cell_methods"] = "time: mean"
        return daily.to_dataset().load()
