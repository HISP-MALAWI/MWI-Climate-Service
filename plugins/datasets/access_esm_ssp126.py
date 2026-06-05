"""ACCESS-ESM1-5 SSP1-2.6 climate projections plugin.

Daily climate variables for Malawi from locally-stored NetCDF files
produced by the ACCESS-ESM1-5 global climate model under the SSP1-2.6
(low-emission) scenario.

Variables:
  tavg  Mean temperature     (°C)
  tmin  Minimum temperature  (°C)
  tmax  Maximum temperature  (°C)
  pre   Precipitation        (mm)

Period: 1970-01-02 – 2101-01-03
Resolution: 0.5° × 0.5° (~55 km)
Projection: geographic (EPSG:4326)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming.protocol import GridSpec

logger = logging.getLogger(__name__)

_RES_DEG = 0.5


class AccessEsmSsp126Plugin:
    """IngestionPlugin for locally-stored ACCESS-ESM1-5 SSP1-2.6 NetCDF files.

    One instance per variable/file. Opens the file lazily on first access
    and slices one day at a time. No network access required.
    """

    max_concurrency = 1
    commit_batch_size = 30
    rechunk_time = 365
    pyramid: bool = True

    def __init__(self, file_path: str, variable: str) -> None:
        self._file_path = file_path
        self._variable = variable
        self._ds: xr.Dataset | None = None

    def _open(self) -> xr.Dataset:
        if self._ds is None:
            logger.info("Opening ACCESS-ESM1-5 SSP1-2.6 file: %s", self._file_path)
            self._ds = xr.open_dataset(self._file_path, chunks={})
        return self._ds

    async def probe(self, bbox: list[float], **_: Any) -> GridSpec:
        xmin, ymin, xmax, ymax = map(float, bbox)
        nx = max(1, round((xmax - xmin) / _RES_DEG))
        ny = max(1, round((ymax - ymin) / _RES_DEG))
        return GridSpec(
            shape=(ny, nx),
            crs=4326,
            dtype=np.dtype("float32"),
            nodata=float("nan"),
            time_dim="t",
            x_dim="x",
            y_dim="y",
        )

    async def periods(self, start: str, end: str) -> list[str]:
        ds = self._open()
        times = ds.time.values
        start_dt = np.datetime64(start[:10], "D")
        end_dt = np.datetime64(end[:10], "D")
        mask = (times.astype("datetime64[D]") >= start_dt) & (
            times.astype("datetime64[D]") <= end_dt
        )
        return [str(t)[:10] for t in times[mask]]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        ds = self._open()
        xmin, ymin, xmax, ymax = map(float, bbox)

        da = ds[self._variable].sel(time=period_id, method="nearest")
        da = da.sel(
            lon=slice(xmin, xmax),
            lat=slice(ymin, ymax),
        )
        da = da.rename({"lon": "x", "lat": "y"})
        da = da.astype("float32")

        result = da.to_dataset()
        result = result.expand_dims({"t": [np.datetime64(period_id)]})
        result = result.load()
        return result
