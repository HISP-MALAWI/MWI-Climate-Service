"""Streaming plugin for the spell hazard grid API.

Serves dry_spell_*, heatwave_* and cold_spell_* grids from

    GET {base_url}/grid/spells?variable=<variable>

One class serves all 18 templates; the template's ``ingestion.params`` picks the
variable. These grids are summaries (counts, lengths, hazard class), not a
time series, so ``periods()`` returns a single period: the requested ``start``.
That is the period stamped on the stored grid.

ASSUMPTION: the response format of /grid/spells is not documented. This plugin
auto-detects GeoTIFF, NetCDF, or JSON ({lat/lon arrays + 2-D values}). If your
server returns something else, adjust ``_to_dataarray`` only.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import rioxarray  # noqa: F401  # activates the .rio accessor used below
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_LAT_KEYS = ("lat", "latitude", "y")
_LON_KEYS = ("lon", "longitude", "x")
_VAL_KEYS = ("values", "data", "grid", "z")


class SpellHazardPlugin(BaseDatasetPlugin):
    max_concurrency = 1  # one grid per ingestion; be gentle on the source
    crs = 4326

    async def periods(self, start: str, end: str) -> list[str]:
        # Single snapshot. Return exactly one id inside the requested window,
        # otherwise the framework refuses the ingestion.
        return [start[:10]]

    def fetch_period(self, period_id: str, bbox: list[float], **params) -> xr.Dataset:
        base_url = params["base_url"].rstrip("/")
        variable = params["source_variable"]
        timeout = float(params.get("timeout", 120))

        url = f"{base_url}/grid/spells?" + urllib.parse.urlencode({"variable": variable})
        raw, ctype = self._download(url, timeout)
        da = self._to_dataarray(raw, ctype, variable)
        if da.rio.crs is None:
            da = da.rio.write_crs(self.crs)
        return normalize_period(
            da.astype("float32"), variable=variable, period=period_id, bbox=bbox
        )

    # ------------------------------------------------------------------ http
    @staticmethod
    def _download(url: str, timeout: float) -> tuple[bytes, str]:
        req = urllib.request.Request(url, headers={"accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), (resp.headers.get("content-type") or "").lower()
        except urllib.error.HTTPError as exc:
            # The API answers {"error": "Variable not found", "available_variables": [...]}
            body = exc.read()[:500].decode("utf-8", "replace")
            raise RuntimeError(f"{url} -> HTTP {exc.code}: {body}") from exc

    # --------------------------------------------------------------- parsing
    def _to_dataarray(self, raw: bytes, ctype: str, name: str) -> xr.DataArray:
        head = raw.lstrip()[:1]
        if "json" in ctype or head in (b"{", b"["):
            return self._from_json(json.loads(raw), name)
        if "tif" in ctype or raw[:4] in (b"II*\x00", b"MM\x00*"):
            return self._from_geotiff(raw)
        if "netcdf" in ctype or raw[:3] == b"CDF" or raw[:4] == b"\x89HDF":
            return self._from_netcdf(raw, name)
        raise ValueError(f"Unrecognised response (content-type={ctype!r}, first bytes={raw[:16]!r})")

    @staticmethod
    def _from_geotiff(raw: bytes) -> xr.DataArray:
        import rioxarray
        from rasterio.io import MemoryFile

        with MemoryFile(raw) as mem, mem.open() as src:
            return rioxarray.open_rasterio(src, masked=True).load()

    @staticmethod
    def _from_netcdf(raw: bytes, name: str) -> xr.DataArray:
        ds = xr.open_dataset(io.BytesIO(raw)).load()
        var = name if name in ds.data_vars else next(iter(ds.data_vars))
        return ds[var]

    @staticmethod
    def _from_json(payload, name: str) -> xr.DataArray:
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"API error: {payload['error']}")
        if not isinstance(payload, dict):
            raise ValueError("JSON array responses are not supported; expected an object")

        pick = lambda keys: next((payload[k] for k in keys if k in payload), None)  # noqa: E731
        lat, lon = pick(_LAT_KEYS), pick(_LON_KEYS)
        vals = payload.get(name, pick(_VAL_KEYS))
        if lat is None or lon is None or vals is None:
            raise ValueError(f"Cannot locate lat/lon/values in JSON; keys = {sorted(payload)}")

        arr = np.asarray(vals, dtype="float64")  # nulls -> NaN
        if arr.shape != (len(lat), len(lon)):
            raise ValueError(f"values shape {arr.shape} != (len(lat), len(lon)) = ({len(lat)}, {len(lon)})")
        return xr.DataArray(arr, dims=("y", "x"), coords={"y": np.asarray(lat), "x": np.asarray(lon)}, name=name)
