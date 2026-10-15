"""Daily gridded forecast plugin (DCCMS grid/forecast API).

Fetches daily variable forecasts (tmin, tmax, etc.) from the DCCMS gridded
forecast service. The upstream API is keyed by forecast *lead day*
(day=1, 2, 3, ...), and each response carries the `date` it predicts.

Stored as one run per issue date, with the lead days on `lead_time` (see
`dccms_runs`), so each sync appends the new run and the dataset can be
scheduled. The issue date is the day before lead day 1's `date`. A sync
before DCCMS publishes the day's forecast therefore finds the run already
stored and writes nothing, rather than storing yesterday's run as today's.

Grid is a near-regular ~7km (0.0629 lat x 0.0645 lon degree) curvilinear
mesh returned as parallel 2D lat/lon/values arrays. The row-to-row and
column-to-column drift is small relative to the ~7km cell size, so a 1D
lat/lon axis is derived (row-mean lat, column-mean lon) to fit the
regular-grid shape normalize_period/BaseDatasetPlugin expect -- this is an
approximation, not an exact reprojection.

Source: Department of Climate Change and Meteorological Services (DCCMS).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any

import numpy as np
import requests
import xarray as xr

from open_climate_service.shared import forecast
from open_climate_service.streaming import BaseDatasetPlugin
from rasters.dccms_runs import forecast_run

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FORECAST_DAYS = 10
_TIMEOUT = 30
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


class DailyForecastPlugin(BaseDatasetPlugin):
    """Streaming plugin for DCCMS daily gridded forecasts (grid/forecast).

    One period is one run, identified by its issue date, with lead days
    1..max_forecast_days on `lead_time`. The template must set
    `temporal_direction: future`.
    """

    max_concurrency = 1
    commit_batch_size = 1
    # The append axis: one run per issue date, so a sync never rewrites an earlier run.
    time_dim = forecast.REFERENCE_DIM

    def __init__(
        self,
        base_url: str,
        dataset: str,
        max_forecast_days: int = _DEFAULT_MAX_FORECAST_DAYS,
        **_: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._variable = dataset
        self._max_forecast_days = max_forecast_days
        # issue date (ISO str) -> the run's payloads by lead day, filled by
        # periods() so fetch_period() does not have to hit the API again.
        self._runs: dict[str, dict[int, dict[str, Any]]] = {}

    def _request_with_retry(self, day: int) -> requests.Response | None:
        endpoint = f"{self._base_url}/grid/forecast"
        params = {"variable": self._variable, "day": day}

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = requests.get(endpoint, params=params, timeout=_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == _MAX_RETRIES:
                    logger.warning(
                        "Giving up on day=%d after %d attempts (%s)", day, attempt, exc
                    )
                    return None
                sleep_for = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Network error on attempt %d/%d for day=%d (%s); retrying in %.1fs",
                    attempt, _MAX_RETRIES, day, exc, sleep_for,
                )
                time.sleep(sleep_for)
                continue

            if response.status_code == 200:
                return response

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                sleep_for = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable status %d on attempt %d/%d for day=%d; retrying in %.1fs",
                    response.status_code, attempt, _MAX_RETRIES, day, sleep_for,
                )
                time.sleep(sleep_for)
                continue

            # Non-retryable (e.g. 404 = lead day not available) or retries exhausted.
            return response

        return None

    def _scan(self) -> tuple[date, dict[int, dict[str, Any]]] | None:
        """The current run: its issue date and its grids by lead day, or None if the API serves none.

        The issue date is the day before lead day 1's `date`; each lead is
        counted from it, so the stored valid dates are the API's own.
        """
        by_date: dict[date, dict[str, Any]] = {}
        for day in range(1, self._max_forecast_days + 1):
            response = self._request_with_retry(day)
            if response is None or response.status_code != 200:
                logger.info(
                    "Stopping forecast lead-day scan for %s at day=%d (%s)",
                    self._variable, day,
                    "no response" if response is None else response.status_code,
                )
                break
            payload = response.json()
            predicted = payload.get("date")
            if not predicted:
                logger.warning("day=%d response missing 'date'; skipping", day)
                continue
            by_date[date.fromisoformat(predicted)] = payload
        if not by_date:
            return None
        issue = min(by_date) - timedelta(days=1)
        return issue, {(when - issue).days: payload for when, payload in by_date.items()}

    async def periods(self, start: str, end: str) -> list[str]:
        """The current run's issue date, when it falls in [start, end]."""
        scanned = await asyncio.to_thread(self._scan)
        if scanned is None:
            return []
        issue, payloads = scanned
        if not date.fromisoformat(start[:10]) <= issue <= date.fromisoformat(end[:10]):
            return []
        self._runs[issue.isoformat()] = payloads
        return [issue.isoformat()]

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """The run issued on `period_id`, as `(reference_time=1, lead_time, y, x)`.

        Only the current run is available upstream; any other issue date
        raises, which aborts the ingest rather than storing a different run
        under its date.
        """
        payloads = self._runs.get(period_id[:10])
        if payloads is None:
            scanned = self._scan()
            if scanned is None or scanned[0].isoformat() != period_id[:10]:
                raise RuntimeError(
                    f"No forecast run issued on {period_id} is available for variable={self._variable}"
                )
            payloads = scanned[1]

        leads = sorted(payloads)
        grids = [self._grid(payloads[day], bbox) for day in leads]
        return forecast_run(grids, leads, date.fromisoformat(period_id[:10]), variable=self._variable).load()

    def _grid(self, payload: dict[str, Any], bbox: list[float]) -> xr.DataArray:
        """One lead day's grid, clipped to `bbox`, as `(y, x)`."""
        xmin, ymin, xmax, ymax = map(float, bbox)
        lat2d = np.asarray(payload["lat"], dtype="float64")
        lon2d = np.asarray(payload["lon"], dtype="float64")
        values = np.asarray(payload["values"], dtype="float32")

        # Curvilinear mesh with small drift: row-mean lat and column-mean lon
        # give the 1D axes a regular grid needs (see the module docstring).
        lat_1d = lat2d.mean(axis=1)
        lon_1d = lon2d.mean(axis=0)

        da = xr.DataArray(
            values,
            dims=("lat", "lon"),
            coords={"lat": lat_1d, "lon": lon_1d},
        )

        lat_slice = (
            slice(ymax, ymin) if da.lat.values[0] > da.lat.values[-1] else slice(ymin, ymax)
        )
        lon_slice = (
            slice(xmax, xmin) if da.lon.values[0] > da.lon.values[-1] else slice(xmin, xmax)
        )
        da = da.sel(lat=lat_slice, lon=lon_slice)

        da = da.rename({"lon": self.x_dim, "lat": self.y_dim})
        da = da.astype("float32")
        if units := payload.get("units"):
            da.attrs["units"] = units
        return da
