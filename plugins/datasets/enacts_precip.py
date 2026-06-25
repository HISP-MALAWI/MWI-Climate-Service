from __future__ import annotations

import asyncio
from datetime import datetime
import io
import logging
import os
from typing import Any

import numpy as np
import requests
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

_OCS_VAR = "precip"
_API_VAR = "precip"
_RES_DEG = 0.05


class EnactsPrecipPlugin(BaseDatasetPlugin):
    """IngestionPlugin for remote ENACTS precipitation data fetched via the DST API.

    Queries the REST API dynamically, pulling down spatial subsets on demand.
    Sources authentication credentials securely from the environment.
    """

    max_concurrency = 1
    commit_batch_size = 30
    rechunk_time = 365
    pyramid: bool = True

    def __init__(
        self, 
        base_url: str, 
        dataset: str = "MON", 
        temporal_resolution: str = "daily",
        api_key: str | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dataset = dataset
        self._temporal_resolution = temporal_resolution
        
        resolved_key = api_key or os.getenv("ENACTS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ENACTS API key missing. Please set 'ENACTS_API_KEY' in your .env file."
            )
            
        self._headers = {
            "Authorization": f"Apikey {resolved_key}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    async def periods(self, start: str, end: str) -> list[str]:
        start_dt = np.datetime64(start[:10], "D")
        end_dt = np.datetime64(end[:10], "D")
        days = np.arange(start_dt, end_dt + np.timedelta64(1, "D"), dtype="datetime64[D]")
        return [str(d) for d in days]

    def _fetch_api_data(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        xmin, ymin, xmax, ymax = map(float, bbox)
        endpoint = f"{self._base_url}/download_raw_data"
        
        dt = datetime.strptime(period_id[:10], "%Y-%m-%d")
        unpadded_date = f"{dt.year}-{dt.month}-{dt.day}"
        
        params = {
            "dataset": self._dataset,
            "temporalRes": self._temporal_resolution,
            "variable": _API_VAR,
            "geomExtract": "rectangle",
            "minLon": xmin,
            "maxLon": xmax,
            "minLat": ymin,
            "maxLat": ymax,
            "Date": unpadded_date,
            "outFormat": "netCDF-Format"
        }

        response = requests.get(endpoint, params=params, headers=self._headers, timeout=45)
        
        if response.status_code != 200:
            error_msg = (
                f"\n=== ENACTS SERVER ERROR ({response.status_code}) ===\n"
                f"URL Called: {response.url}\n"
                f"Server Response Body: {response.text}\n"
                f"========================================="
            )
            logger.error(error_msg)
            raise requests.HTTPError(error_msg, response=response)

        with xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf") as ds:
            rename_coords = {}
            if "Time" in ds.variables or "Time" in ds.dims: rename_coords["Time"] = "time"
            if "Lat" in ds.variables or "Lat" in ds.dims: rename_coords["Lat"] = "lat"
            if "Lon" in ds.variables or "Lon" in ds.dims: rename_coords["Lon"] = "lon"
            if "longitude" in ds.variables: rename_coords["longitude"] = "lon"
            if "latitude" in ds.variables: rename_coords["latitude"] = "lat"
            
            if rename_coords:
                ds = ds.rename(rename_coords)

            if "rr" in ds.data_vars:
                var_key = "rr"
            elif "precip" in ds.data_vars:
                var_key = "precip"
            else:
                var_key = list(ds.data_vars)[0]

            da = ds[var_key].sel(time=period_id, method="nearest")
            da = da.sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
            
            da = da.rename({"lon": "x", "lat": "y"})
            da = da.astype("float32")
            da.attrs["units"] = "mm"

            if "time" in da.coords:
                da = da.drop_vars("time")

            result = da.rename(_OCS_VAR).to_dataset()
            result = result.expand_dims(time=[np.datetime64(period_id)])
            return result.load()

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        return await asyncio.to_thread(self._fetch_api_data, period_id, bbox)