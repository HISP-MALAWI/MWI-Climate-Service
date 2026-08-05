"""Daily gridded forecast plugin (DCCMS grid/forecast API).

Fetches daily variable forecasts (tmin, tmax, etc.) from the DCCMS gridded
forecast service. The upstream API is keyed by forecast *lead day*
(day=1, 2, 3, ...) rather than a calendar date range, so `periods()`
ignores start/end and instead walks lead days until the API stops
returning data.

Grid is a near-regular ~7km (0.0629 lat x 0.0645 lon degree) curvilinear
mesh returned as parallel 2D lat/lon/values arrays. The row-to-row and
column-to-column drift is small relative to the ~7km cell size, so a 1D
lat/lon axis is derived (row-mean lat, column-mean lon) to fit the
regular-grid shape normalize_period/BaseDatasetPlugin expect -- this is an
approximation, not an exact reprojection.

Source: Department of Climate Change and Meteorological Services (DCCMS).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import requests
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FORECAST_DAYS = 10
_TIMEOUT = 30
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


class DailyForecastPlugin(BaseDatasetPlugin):
    """Streaming plugin for DCCMS daily gridded forecasts (grid/forecast).

    Subclasses BaseDatasetPlugin, so `time_dim`/`x_dim`/`y_dim`/`crs`
    inherit the framework defaults ("t"/"x"/"y"/4326).

    Ignores the dataset's period_type/resolution config entirely --
    periods are lead days (1..max_forecast_days) from the API, not a
    calendar range, so `start`/`end` passed into periods() are unused.
    """

    max_concurrency = 1
    commit_batch_size = 1

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
        # period_id (ISO date str) -> raw API payload, populated during
        # periods() so fetch_period() doesn't have to re-hit the API.
        self._cache: dict[str, dict[str, Any]] = {}

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

    async def periods(self, start: str, end: str) -> list[str]:
        # start/end intentionally ignored -- this API is lead-day based,
        # not a calendar range. Walk day=1.. until the API stops
        # returning 200, capped at max_forecast_days.
        periods: list[str] = []
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
            period_id = payload.get("date")
            if not period_id:
                logger.warning("day=%d response missing 'date'; skipping", day)
                continue
            self._cache[period_id] = payload
            periods.append(period_id)
        return periods

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Build a one-step dataset for the given forecast date.

        A regular (blocking) method -- the framework runs it in a worker
        thread, matching how EnactsPrecipPlugin's fetch_period behaves.
        Raises if the day isn't available, which aborts the ingest rather
        than silently skipping (same convention as EnactsPrecipPlugin).
        """
        xmin, ymin, xmax, ymax = map(float, bbox)

        payload = self._cache.pop(period_id, None)
        if payload is None:
            # Standalone call without a prior periods() pass -- look the
            # date up by scanning lead days again.
            for day in range(1, self._max_forecast_days + 1):
                response = self._request_with_retry(day)
                if response is None or response.status_code != 200:
                    break
                candidate = response.json()
                if candidate.get("date") == period_id:
                    payload = candidate
                    break

        if payload is None:
            raise RuntimeError(
                f"No forecast data available for variable={self._variable} on {period_id}"
            )

        lat2d = np.asarray(payload["lat"], dtype="float64")
        lon2d = np.asarray(payload["lon"], dtype="float64")
        values = np.asarray(payload["values"], dtype="float32")

        # Grid is curvilinear but near-regular at this resolution: collapse
        # to 1D axes (row-mean lat, column-mean lon) so it fits the
        # regular x/y grid normalize_period expects.
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
        da.attrs["units"] = payload.get("units", "")

        return normalize_period(da, variable=self._variable, period=period_id).load()