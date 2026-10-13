"""ENACTS daily precip/tmin/tmax plugin.

Fetches daily precipitation, minimum-temperature, or maximum-temperature
rasters from the DCCMS ENACTS DST API, one calendar day per request,
clipped server-side to the requested bbox. Which variable is fetched is
controlled by the `variable` constructor param.
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime

import numpy as np
import requests
import xarray as xr

from open_climate_service.shared.time import utc_today
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

logger = logging.getLogger(__name__)

# "MON" is the fixed dataset code for this ENACTS source (not an
# abbreviation for "monthly") -- periodicity is controlled separately via
# the temporalRes param, which is set to "daily" below.
_KNOWN_DAILY_DATASET_CODES = {"MON"}

# Per-variable lookup: candidate data_var names to look for in the
# returned netCDF (tried in order -- server-side naming isn't fully
# consistent across variables, e.g. "rr" vs "precip"), and the CF units
# string to stamp onto the output DataArray.
#
# NOTE: the tmin/tmax var_keys are best-guess based on the precip entry's
# pattern (short WMO-style code + long-form fallback) -- confirm against a
# live API response before relying on these for tmin/tmax.
_VAR_KEY_CANDIDATES: dict[str, list[str]] = {
    "precip": ["rr", "precip"],
    "tmin": ["tn", "tmin"],
    "tmax": ["tx", "tmax"],
}
_UNITS: dict[str, str] = {
    "precip": "mm",
    "tmin": "degC",
    "tmax": "degC",
}

# Response bodies logged on error are truncated to this many characters so a
# large HTML/error page doesn't flood the logs.
_MAX_LOGGED_BODY_CHARS = 2000

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


class EnactsPlugin(BaseDatasetPlugin):
    """Streaming plugin for ENACTS daily variables (DCCMS DST API).

    Handles precipitation, tmin, and tmax via a single implementation --
    which of the three is fetched is controlled by the `variable`
    constructor param.

    Subclasses BaseDatasetPlugin, so `time_dim`/`x_dim`/`y_dim`/`crs`
    inherit the framework defaults ("t"/"x"/"y"/4326) rather than being
    redeclared -- the grid is inferred by the orchestrator from the first
    fetched period.
    """

    max_concurrency = 1
    commit_batch_size = 30

    def __init__(
        self,
        base_url: str,
        variable: str,
        dataset: str = "MON",
        temporal_resolution: str = "daily",
        api_key: str | None = None,
        **_: object,
    ) -> None:
        if variable not in _VAR_KEY_CANDIDATES:
            raise ValueError(
                f"Unknown ENACTS variable '{variable}'. "
                f"Expected one of {sorted(_VAR_KEY_CANDIDATES)}."
            )

        self._base_url = base_url.rstrip("/")
        self._dataset = dataset
        self._variable = variable
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
            ),
        }

    async def periods(self, start: str, end: str) -> list[str]:
        start_dt = np.datetime64(start[:10], "D")
        end_dt = np.datetime64(end[:10], "D")
        days = np.arange(start_dt, end_dt + np.timedelta64(1, "D"), dtype="datetime64[D]")
        return [str(d) for d in days]

    def _request_with_retry(self, endpoint: str, params: dict) -> requests.Response:
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = requests.get(
                    endpoint, params=params, headers=self._headers, timeout=45
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES:
                    raise
                sleep_for = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Network error on attempt %d/%d for %s (%s); retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    endpoint,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            if response.status_code == 200:
                return response

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                sleep_for = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable status %d on attempt %d/%d for %s; retrying in %.1fs",
                    response.status_code,
                    attempt,
                    _MAX_RETRIES,
                    endpoint,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            # Non-retryable status (e.g. 400/401/404), or retries exhausted.
            return response

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Exhausted retries without a response or exception.")

    def fetch_period(self, period_id: str, bbox: list[float], **_: object) -> xr.Dataset:
        """Fetch one day's raster, clip it to the bbox, and return a one-step dataset.

        A regular (blocking) method -- the framework runs it in a worker
        thread. Raises if the day is unavailable, mirroring how other
        plugins in this codebase (e.g. CHIRPS3) signal "not published": the
        orchestrator does not catch per-period exceptions, so this aborts
        the ingest rather than silently skipping.
        """
        xmin, ymin, xmax, ymax = map(float, bbox)
        endpoint = f"{self._base_url}/download_raw_data"

        dt = datetime.strptime(period_id[:10], "%Y-%m-%d")
        unpadded_date = f"{dt.year}-{dt.month}-{dt.day}"

        params = {
            "dataset": self._dataset,
            "temporalRes": self._temporal_resolution,
            "variable": self._variable,
            "geomExtract": "rectangle",
            "minLon": xmin,
            "maxLon": xmax,
            "minLat": ymin,
            "maxLat": ymax,
            "Date": unpadded_date,
            "outFormat": "netCDF-Format",
        }

        response = self._request_with_retry(endpoint, params)

        if response.status_code == 404:
            raise RuntimeError(f"No ENACTS data available for {period_id} (404): {endpoint}")

        if response.status_code != 200:
            body = response.text[:_MAX_LOGGED_BODY_CHARS]
            error_msg = (
                f"\n=== ENACTS SERVER ERROR ({response.status_code}) ===\n"
                f"URL Called: {response.url}\n"
                f"Server Response Body (truncated): {body}\n"
                f"========================================="
            )
            logger.error(error_msg)
            raise requests.HTTPError(error_msg, response=response)

        with xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf") as ds:
            # The API's netCDF uses capitalized coord names, which don't
            # match normalize_period's lowercase match lists (lon/lat/time),
            # so lowercase them here before handing off.
            rename_map = {
                "Time": "time",
                "Lat": "lat",
                "Lon": "lon",
                "longitude": "lon",
                "latitude": "lat",
            }
            rename_coords = {
                src: dst
                for src, dst in rename_map.items()
                if src in ds.variables or src in ds.dims
            }
            if rename_coords:
                ds = ds.rename(rename_coords)

            candidates = _VAR_KEY_CANDIDATES[self._variable]
            var_key = next((key for key in candidates if key in ds.data_vars), None)
            if var_key is None:
                var_key = list(ds.data_vars)[0]
                logger.warning(
                    "None of expected data_vars %s found for variable '%s' "
                    "in ENACTS response for %s; falling back to first "
                    "data_var '%s'.",
                    candidates,
                    self._variable,
                    period_id,
                    var_key,
                )

            if "time" in ds.dims and ds.sizes["time"] > 1:
                # Bound the "nearest" match so a gap in the record raises
                # instead of silently mislabeling a different day's data.
                da = ds[var_key].sel(
                    time=period_id,
                    method="nearest",
                    tolerance=np.timedelta64(1, "D"),
                )
            else:
                da = ds[var_key].isel(time=0) if "time" in ds.dims else ds[var_key]

            # The server already clips to the requested bbox via
            # minLon/maxLon/minLat/maxLat, but re-slice defensively client
            # side too. Coordinate direction can vary independently on each
            # axis, so check and handle lat and lon separately rather than
            # assuming longitude is always ascending.
            lat_slice = (
                slice(ymax, ymin) if da.lat.values[0] > da.lat.values[-1] else slice(ymin, ymax)
            )
            lon_slice = (
                slice(xmax, xmin) if da.lon.values[0] > da.lon.values[-1] else slice(xmin, xmax)
            )
            da = da.sel(lon=lon_slice, lat=lat_slice)

            da = da.rename({"lon": self.x_dim, "lat": self.y_dim})
            da = da.astype("float32")
            da.attrs["units"] = _UNITS[self._variable]

            if "time" in da.coords:
                da = da.drop_vars("time")

            # normalize_period wraps the DataArray into a Dataset named
            # after `variable` and stamps period_id onto self.time_dim
            # ("t"). No bbox= is passed since the array is already clipped
            # above (and normalize_period's bbox clip needs a CRS-aware rio
            # accessor this array doesn't carry).
            return normalize_period(da, variable=self._variable, period=period_id).load()