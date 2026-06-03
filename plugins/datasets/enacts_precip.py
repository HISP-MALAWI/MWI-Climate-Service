from __future__ import annotations

import asyncio
import io
import logging
import os  # Added for environment variable access
from typing import Any

import numpy as np
import requests
import xarray as xr

from open_climate_service.streaming.protocol import GridSpec

logger = logging.getLogger(__name__)

_VAR = "precip"
_RES_DEG = 0.05


class EnactsPrecipPlugin:
    """IngestionPlugin for remote ENACTS precipitation data fetched via the DST API.

    Queries the REST API dynamically, sourcing authentication securely from the environment.
    """

    max_concurrency = 4
    commit_batch_size = 30
    rechunk_time = 365
    pyramid: bool = True

    def __init__(
        self, 
        base_url: str, 
        dataset: str = "MON", 
        temporal_resolution: str = "daily",
        api_key: str | None = None  # Now optional in YAML
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dataset = dataset
        self._temporal_resolution = temporal_resolution
        
        # Fallback to os.getenv if not explicitly provided in the YAML block
        resolved_key = api_key or os.getenv("ENACTS_API_KEY")
        
        if not resolved_key:
            raise ValueError(
                "ENACTS API key missing. Please set 'ENACTS_API_KEY' in your .env file "
                "or pass 'api_key' inside the plugin parameters."
            )
            
        self._headers = {"Authorization": f"Apikey {resolved_key}"}

    async def probe(self, bbox: list[float], **_: Any) -> GridSpec:
        xmin, ymin, xmax, ymax = map(float, bbox)
        nx = max(1, round((xmax - xmin) / _RES_DEG))
        ny = max(1, round((ymax - ymin) / _RES_DEG))
        
        return GridSpec(
            shape=(ny, nx),
            crs=4326,
            dtype=np.dtype("float32"),
            nodata=float("nan"),
            time_dim="time",
        )

    async def periods(self, start: str, end: str) -> list[str]:
        start_dt = np.datetime64(start[:10], "D")
        end_dt = np.datetime64(end[:10], "D")
        days = np.arange(start_dt, end_dt + np.timedelta64(1, "D"), dtype="datetime64[D]")
        return [str(d) for d in days]

    def _fetch_api_data(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        xmin, ymin, xmax, ymax = map(float, bbox)
        endpoint = f"{self._base_url}/download_raw_data"
        
        params = {
            "dataset": self._dataset,
            "temporal_resolution": self._temporal_resolution,
            "variable": _VAR,
            "geomExtract": "rectangle",
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "start_date": period_id,
            "end_date": period_id,
            "format": "netcdf"
        }

        logger.info("Requesting ENACTS Precip for %s over bounds %s", period_id, bbox)
        response = requests.get(endpoint, params=params, headers=self._headers, timeout=30)
        response.raise_for_status()

        with xr.open_dataset(io.BytesIO(response.content)) as ds:
            lon_key = "lon" if "lon" in ds.coords else "longitude"
            lat_key = "lat" if "lat" in ds.coords else "latitude"
            var_key = _VAR if _VAR in ds.data_vars else list(ds.data_vars)[0]

            da = ds[var_key].sel(time=period_id, method="nearest")
            da = da.sel({lon_key: slice(xmin, xmax), lat_key: slice(ymin, ymax)})
            
            da = da.rename({lon_key: "x", lat_key: "y"})
            da = da.astype("float32")
            da.attrs["units"] = "mm"

            result = da.rename(_VAR).to_dataset()
            result = result.expand_dims(time=[np.datetime64(period_id)])
            return result.load()

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        return await asyncio.to_thread(self._fetch_api_data, period_id, bbox)