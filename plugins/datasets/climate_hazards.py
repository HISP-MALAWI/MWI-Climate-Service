"""Daily gridded hazard plugin (DCCMS grid/hazards API).

Fetches daily hazard classification grids (e.g. rain_hazard) from the
DCCMS gridded hazard service. Sibling to DailyForecastPlugin
(datasets.daily_forecast), but NOT a subclass of it -- the two upstream
endpoints differ enough that sharing a base class would mean overriding
most of it anyway:

  * Endpoint is `/grid/hazards` (forecast's is `/grid/forecast`).
  * Lead days are 0-indexed (day=0 is the current/issue day), not
    1-indexed like /grid/forecast.
  * The response has no `date` field -- forecast derives period_id
    directly from the upstream payload; here it must be computed as
    `today + day` days, since the API only tells us the lead-day offset.
  * `lat`/`lon` are already flat 1D axes matching `shape`, not a 2D
    curvilinear mesh needing row/column averaging.

Per dhis2/open-climate-service#332, this dataset's template must declare
`temporal_direction: future` so core resolves an omitted `start` to "now"
and an omitted `end` to a generous forward horizon -- this plugin never
receives `None` for either, but it must clip its own output to `end`
itself (and skip anything before `start`), since core rejects a plugin
whose materialized periods fall outside the requested scope
("Materialized artifact coverage does not match the requested scope").

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

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FORECAST_DAYS = 7
_TIMEOUT = 30
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0
_REQUIRED_GRID_KEYS = ("lat", "lon", "values")


def _is_valid_grid_payload(payload: dict[str, Any]) -> bool:
    """True if `payload` looks like a real grid response.

    The API can return HTTP 200 for a lead day it can't actually serve
    (e.g. a day beyond this variable's real forecast horizon), with a
    body that isn't a grid -- so status_code alone isn't a reliable
    signal that a day is usable. Checking for these keys is cheap and
    turns what would otherwise be a bare KeyError deep in fetch_period()
    into a clean "day not available" skip here instead.
    """
    return all(key in payload for key in _REQUIRED_GRID_KEYS)


class HazardForecastPlugin(BaseDatasetPlugin):
    """Streaming plugin for DCCMS daily gridded hazard classes (grid/hazards).

    Subclasses BaseDatasetPlugin directly, so `time_dim`/`x_dim`/`y_dim`/`crs`
    inherit the framework defaults ("t"/"x"/"y"/4326).

    Ignores the dataset's period_type/resolution config -- periods are
    lead days (0..max_forecast_days-1) from the API, not a calendar range.
    `start`/`end` passed into periods() ARE respected: lead days whose
    computed calendar date falls outside that window are filtered out.
    The dataset template must set `temporal_direction: future` so core
    resolves an omitted start to "now" rather than requiring one.
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
        endpoint = f"{self._base_url}/grid/hazards"
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
        # Per dhis2/open-climate-service#332, core never passes None here --
        # an omitted start resolves to "now" and an omitted end resolves to
        # a generous forward horizon. Both are always concrete ISO date
        # strings. The plugin clips its own lead-day walk to `end` and
        # skips anything before `start`; core rejects materialized coverage
        # outside the requested window.
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        # Lead day 0 is "today" -- the API gives no date field, so the
        # calendar date for each lead day has to be computed relative to
        # the current issue day rather than read off the payload.
        issue_date = date.today()

        periods: list[str] = []
        for day in range(0, self._max_forecast_days):
            response = await asyncio.to_thread(self._request_with_retry, day)
            if response is None or response.status_code != 200:
                logger.info(
                    "Stopping hazard lead-day scan for %s at day=%d (%s)",
                    self._variable, day,
                    "no response" if response is None else response.status_code,
                )
                break
            payload = response.json()
            if not _is_valid_grid_payload(payload):
                logger.warning(
                    "day=%d response for %s missing lat/lon/values; skipping",
                    day, self._variable,
                )
                continue

            period_date = issue_date + timedelta(days=day)
            if period_date > end_date:
                # Stop rather than skip: lead days are strictly increasing,
                # so nothing further in the walk can be <= end either.
                logger.info(
                    "Stopping hazard lead-day scan for %s at day=%d: "
                    "%s exceeds requested end=%s",
                    self._variable, day, period_date.isoformat(), end,
                )
                break
            if period_date < start_date:
                logger.debug(
                    "Skipping %s day=%d: %s is before requested start=%s",
                    self._variable, day, period_date.isoformat(), start,
                )
                continue

            period_id = period_date.isoformat()
            self._cache[period_id] = payload
            periods.append(period_id)
        return periods

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Build a one-step dataset for the given hazard date.

        A regular (blocking) method -- the framework runs it in a worker
        thread, matching DailyForecastPlugin.fetch_period. Raises if the
        day isn't available, which aborts the ingest rather than silently
        skipping (same convention as DailyForecastPlugin).
        """
        xmin, ymin, xmax, ymax = map(float, bbox)

        payload = self._cache.get(period_id, None)
        if payload is None:
            # Standalone call without a prior periods() pass -- look the
            # date up by recomputing each lead day's calendar date the
            # same way periods() does, and stopping at the one that matches.
            issue_date = date.today()
            for day in range(0, self._max_forecast_days):
                if (issue_date + timedelta(days=day)).isoformat() != period_id:
                    continue
                response = self._request_with_retry(day)
                if response is not None and response.status_code == 200:
                    candidate = response.json()
                    if _is_valid_grid_payload(candidate):
                        payload = candidate
                break

        if payload is None:
            raise RuntimeError(
                f"No hazard data available for variable={self._variable} on {period_id}"
            )

        # Unlike DailyForecastPlugin's payload, lat/lon here are already
        # flat 1D axes matching `shape` -- no curvilinear row/column
        # averaging needed.
        lat_1d = np.asarray(payload["lat"], dtype="float64")
        lon_1d = np.asarray(payload["lon"], dtype="float64")
        values = np.asarray(payload["values"], dtype="float32")

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

        return normalize_period(da, variable=self._variable, period=period_id).load()