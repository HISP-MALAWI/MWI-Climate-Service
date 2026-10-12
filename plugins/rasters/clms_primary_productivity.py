"""CLMS Primary Production plugin.

10-daily (dekad) GPP and NPP at 300m from the Copernicus Land Monitoring Service.
Source: Copernicus Data Space Ecosystem (CDSE) — requires S3 credentials.

Data is available on the 1st, 11th and 21st of each month from 2014 onwards.

Credentials:
    - Register at https://dataspace.copernicus.eu/
    - Generate S3 keys at https://eodata-s3keysmanager.dataspace.copernicus.eu/
    - Set them in the environment
        CDSE_S3_ACCESS_KEY=<ACCESS-KEY>
        CDSE_S3_SECRET_KEY=<SECRET-KEY>

Docs: https://documentation.dataspace.copernicus.eu/APIs/S3.html
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming.protocol import GridSpec

logger = logging.getLogger(__name__)

# 300m ≈ 1/360 degree
_RES_DEG = 1.0 / 360.0

# Defaults select the GPP product; override via ingestion.params for NPP.
_COLLECTION = "clms_gpp_global_300m_10daily_v2_cog"
_ASSET_KEY = "gpp300_gpp"
_VAR = "gpp"
_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
# Pin the OTC (Amsterdam) backend rather than the default load-balanced host.
# Prevents the InvalidAccessKeyId since the S3 keys are only valid on OTC
_S3_ENDPOINT = "https://eodata.ams.dataspace.copernicus.eu"
_S3_BUCKET = "eodata"
_FIRST_YEAR = 2014

# Dekad day-of-month offsets: 1st, 11th, 21st
_DEKAD_DAYS = (1, 11, 21)

# Physical flag values to mask out of the raw raster, before the CF scale/offset.
# (NPP can legitimately be negative — net of respiration — so we mask only the
# flags, not all sub-zero values.)
_FLAG_MISSING = -1
_FLAG_WATER = -2


def _dekad_dates(start: str, end: str) -> list[str]:
    """Return all dekad dates (1st, 11th, 21st) within [start, end]."""
    d_start = date.fromisoformat(start[:10])
    d_end = date.fromisoformat(end[:10])
    results = []
    year = d_start.year
    month = d_start.month
    while True:
        for day in _DEKAD_DAYS:
            try:
                d = date(year, month, day)
            except ValueError:
                continue
            if d < d_start:
                continue
            if d > d_end:
                return results
            results.append(d.isoformat())
        month += 1
        if month > 12:
            month = 1
            year += 1


class ClmsPrimaryProductivityPlugin:
    """IngestionPlugin for CLMS NPP and GPP 300m 10-daily data via CDSE STAC.

    Fetches Cloud-Optimised GeoTIFFs from the Copernicus Data Space Ecosystem S3 storage, clips to the instance bbox, and commits one timestep per dekad.

    GPP vs NPP is selected entirely by `ingestion.params` (collection, asset_key,
    variable); the fetch logic is identical across both products.
    """

    max_concurrency = 2
    commit_batch_size = 1
    rechunk_time = 30
    pyramid: bool = True

    def __init__(
        self,
        *,
        collection: str = _COLLECTION,
        asset_key: str = _ASSET_KEY,
        variable: str = _VAR,
    ) -> None:
        # Stored on the instance (not just read from per-call params) because the
        # orchestrator calls periods() WITHOUT params — only the constructor,
        # probe() and fetch_period() receive them.
        self.collection = collection
        self.asset_key = asset_key
        self.variable = variable

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
        clamped_start = max(start[:10], f"{_FIRST_YEAR}-01-01")
        latest = await asyncio.to_thread(self._latest_date)
        if latest is None:
            return []
        clamped_end = min(end[:10], latest)
        return _dekad_dates(clamped_start, clamped_end)

    def _latest_date(self) -> str | None:
        """Return the date of the most recent published dekad, or None if empty."""
        import pystac_client

        catalog = pystac_client.Client.open(_STAC_URL)
        search = catalog.search(
            collections=[self.collection], max_items=1, sortby="-datetime"
        )
        items = list(search.items())
        if not items:
            return None
        return items[0].datetime.date().isoformat()

    async def fetch_period(
        self, period_id: str, bbox: list[float], **_: Any
    ) -> xr.Dataset:
        return await asyncio.to_thread(self._fetch_sync, period_id, bbox)

    def _fetch_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        import os

        import boto3
        import pystac_client
        import rasterio
        import rioxarray as rxr
        from rasterio.session import AWSSession

        xmin, ymin, xmax, ymax = map(float, bbox)
        # period_id is a dekad date (1st/11th/21st) from periods()
        day = date.fromisoformat(period_id)

        catalog = pystac_client.Client.open(_STAC_URL)
        search = catalog.search(
            collections=[self.collection],
            bbox=[xmin, ymin, xmax, ymax],
            datetime=f"{day.isoformat()}T00:00:00Z/{day.isoformat()}T23:59:59Z",
            max_items=1,
        )
        items = list(search.items())
        if not items:
            raise ValueError(
                f"No CLMS {self.variable.upper()} item found for dekad {day.isoformat()}"
            )

        access_key = os.environ.get("CDSE_S3_ACCESS_KEY")
        secret_key = os.environ.get("CDSE_S3_SECRET_KEY")
        if not access_key or not secret_key:
            raise RuntimeError(
                "Missing CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY envvars"
            )

        s3_client = boto3.client(
            "s3",
            endpoint_url=_S3_ENDPOINT,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="default",
        )
        href = items[0].assets[self.asset_key].href
        s3_key = self._resolve_object_key(s3_client, href)
        logger.info(
            "Fetching CLMS %s %s: s3://%s/%s",
            self.variable.upper(),
            period_id,
            _S3_BUCKET,
            s3_key,
        )

        # GDAL /vsis3 range requests (COG) — only fetches tiles covering the bbox,
        # not the full global file. Credentials go through a boto3-backed AWSSession
        # (rasterio's only sanctioned way to inject S3 keys; it also sets AWS_S3_ENDPOINT
        # from endpoint_url). AWS_VIRTUAL_HOSTING=FALSE selects path-style addressing,
        # required for the CDSE endpoint. rasterio.Env scopes all of this thread-locally,
        # so concurrent fetches don't clobber each other's settings.
        session = AWSSession(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=_S3_ENDPOINT,
            region_name="default",
        )
        with rasterio.Env(
            session,
            AWS_VIRTUAL_HOSTING="FALSE",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_CHUNK_SIZE="1048576",
        ):
            da = rxr.open_rasterio(
                f"/vsis3/{_S3_BUCKET}/{s3_key}", chunks=None, masked=True, lock=False
            )
            if not isinstance(da, xr.DataArray):
                raise TypeError(
                    f"Expected DataArray from CLMS raster read, got {type(da).__name__}"
                )
            da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
            da = da.squeeze("band", drop=True)
            # Mask flag values (Missing=-1, Water=-2), then apply CF scale factor.
            # NPP can be genuinely negative, so we drop only the flags — not all
            # values below zero, as a `>= 0` clip would.
            da = da.where((da != _FLAG_MISSING) & (da != _FLAG_WATER))
            # apply CF scale/offset by retrieving scale factor and add_offset from either .encoding or .attrs
            scale = float(
                da.attrs.get("scale_factor", da.encoding.get("scale_factor", 1.0))
            )
            offset = float(
                da.attrs.get("add_offset", da.encoding.get("add_offset", 0.0))
            )
            da = (da * scale + offset).astype("float32")
            # Materialize the range reads while the GDAL Env is still active
            da = da.load()

        # Ensure y is ascending (south → north)
        if da.y.values[0] > da.y.values[-1]:
            da = da.isel(y=slice(None, None, -1))

        ds = da.to_dataset(name=self.variable)
        ds = ds.expand_dims({"t": [np.datetime64(day.isoformat(), "D")]})
        # Drop the stray scalar `crs` coord (value 0); the real CRS is carried by
        # the GridSpec from probe(), so this is just clutter in the store.
        ds = ds.drop_vars("crs", errors="ignore")
        return ds

    def _resolve_object_key(self, s3_client: Any, href: str) -> str:
        """Resolve the live S3 key for a STAC asset, ignoring its version path.

        This funtion lists the dekad's folder and matches on the version-independent identity
        (RT level + timestamp + variable), preferring the most recently written file if versions overlap.
        """
        import re

        key = href.removeprefix(f"s3://{_S3_BUCKET}/")
        date_prefix = key.rsplit("/", 2)[0] + "/"  # strip <product-dir>/<file>
        match = re.search(r"RT\d+_\d+", key)  # e.g. RT6_202506100000
        if match is None:
            raise ValueError(f"Unexpected CLMS asset href, cannot resolve key: {href}")
        token = match.group()
        marker = f"-{self.variable.upper()}-"  # -GPP- / -NPP- (excludes QFLAG)
        objs = s3_client.list_objects_v2(Bucket=_S3_BUCKET, Prefix=date_prefix).get(
            "Contents", []
        )
        candidates = [
            o
            for o in objs
            if token in o["Key"]
            and o["Key"].endswith((".tif", ".tiff"))
            and marker in o["Key"]
        ]
        if not candidates:
            raise ValueError(
                f"No {self.variable.upper()} COG under {date_prefix} for {token}"
            )
        return max(candidates, key=lambda o: o["LastModified"])["Key"]
