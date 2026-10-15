"""Daily gridded hazard plugin (DCCMS grid/hazards API).

Fetches daily hazard classification grids (e.g. rain_hazard) from the
DCCMS gridded hazard service. Sibling to DailyForecastPlugin
(rasters.daily_forecast), but NOT a subclass of it -- the two upstreamendpoints differ enough that sharing a base class would mean overriding
most of it anyway:

  * Endpoint is `/grid/hazards` (forecast's is `/grid/forecast`).
  * Lead days are 0-indexed (day=0 is the current/issue day), not
    1-indexed like /grid/forecast.
  * The response has no `date` field -- forecast derives period_id
    directly from the upstream payload; here it must be computed as
    `today + day` days, since the API only tells us the lead-day offset.
  * `lat`/`lon` are already flat 1D axes matching `shape`, not a 2D
    curvilinear mesh needing row/column averaging.

Stored as one run per issue date, with the lead days on `lead_time` (see
`dccms_runs`), so each sync appends the day's run and the dataset can be
scheduled. The API serves only the current run and gives no issue date, so
the run is dated today.

Source: Department of Climate Change and Meteorological Services (DCCMS).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date
from typing import Any

import numpy as np
import requests
import xarray as xr

from open_climate_service.shared import forecast
from open_climate_service.streaming import BaseDatasetPlugin
from rasters.dccms_runs import forecast_run

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

    One period is one run: today's, with lead days 0..max_forecast_days-1
    on `lead_time`. The template must set `temporal_direction: future`.
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

    def _scan(self) -> dict[int, dict[str, Any]]:
        """The current run's grids by lead day, stopping at the first day the API cannot serve."""
        payloads: dict[int, dict[str, Any]] = {}
        for day in range(0, self._max_forecast_days):
            response = self._request_with_retry(day)
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
            payloads[day] = payload
        return payloads

    async def periods(self, start: str, end: str) -> list[str]:
        """Today's run, when it falls in [start, end] and the API serves it."""
        issue = date.today()
        if not date.fromisoformat(start[:10]) <= issue <= date.fromisoformat(end[:10]):
            return []
        payloads = await asyncio.to_thread(self._scan)
        if not payloads:
            return []
        self._runs[issue.isoformat()] = payloads
        return [issue.isoformat()]

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """The run issued on `period_id`, as `(reference_time=1, lead_time, y, x)`.

        Only today's run is available upstream; any other date raises, which
        aborts the ingest rather than storing a different run under its date.
        """
        payloads = self._runs.get(period_id[:10])
        if payloads is None:
            if period_id[:10] != date.today().isoformat():
                raise RuntimeError(
                    f"Only today's hazard run is available for variable={self._variable}, not {period_id}"
                )
            payloads = self._scan()
        if not payloads:
            raise RuntimeError(f"No hazard data available for variable={self._variable} on {period_id}")

        leads = sorted(payloads)
        grids = [self._grid(payloads[day], bbox) for day in leads]
        return forecast_run(grids, leads, date.fromisoformat(period_id[:10]), variable=self._variable).load()

    def _grid(self, payload: dict[str, Any], bbox: list[float]) -> xr.DataArray:
        """One lead day's grid, clipped to `bbox`, as `(y, x)`."""
        xmin, ymin, xmax, ymax = map(float, bbox)
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
        return da
