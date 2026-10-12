"""Streaming plugin for locally supplied seasonal-forecast NetCDF files.

Serves an already-downloaded seasonal forecast (a single NetCDF file, issued
once, covering a handful of monthly lead times) into Open Climate Service.
Unlike most OCS dataset plugins this does not poll a remote API: the whole
file is loaded once, when the plugin is constructed, and served back one
calendar month at a time.

Used by two dataset templates in mwi_seasonal_forecast.yaml — a Tmean
forecast and a rainfall forecast — that share this one class and differ only
in `ingestion.params` (the "multiple templates, one plugin class" pattern
described in the OCS "Adding custom datasets" guide).

Reads the source .nc files straight out of the instance's own
forecast_data/ directory (a sibling of plugins/, next to
climate-service.yaml) rather than bundling copies under the plugin — see
`nc_path` below.
"""

from pathlib import Path

import pandas as pd
import rioxarray  # noqa: F401  # activates the .rio accessor used below
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period


class LocalNetCDFForecastPlugin(BaseDatasetPlugin):
    """Serves one variable out of a local, already-downloaded NetCDF forecast.

    ingestion.params:
        nc_path: path to the .nc file. A relative path (e.g.
            "forecast_data/tmean_seasonal_forecast_2026_2027.nc") is
            resolved against the instance's working directory, the same way
            `data_dir` in climate-service.yaml is — i.e. it assumes the
            service runs from the instance repo root. Use an absolute path
            instead if that doesn't hold for your deployment.
        source_variable: name of the data variable inside the source file
            (e.g. "Tmean", "rainfall_forecast").
        variable: name to store the variable under in OCS. Must match this
            template's top-level `variable:` field.
    """

    # Both source files are plain WGS84 lat/lon grids, but neither carries
    # any CRS metadata of its own. normalize_period's bbox clip needs a CRS
    # attached via rioxarray regardless of whether the grid is projected
    # (see the seNorge/UTM example in the OCS "Adding custom datasets"
    # guide) — without this, fetch_period fails with "CRS not found. Please
    # set the CRS with 'rio.write_crs()'."
    crs = 4326

    def __init__(
        self,
        *,
        nc_path: str,
        source_variable: str,
        variable: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.source_variable = source_variable
        self.variable = variable

        # Load fully into memory up front. fetch_period is called once per
        # period and the framework closes whatever Dataset it returns after
        # writing it, so the per-period slices handed back below must not
        # depend on a live file handle — .load() pulls the backing store
        # into plain in-memory numpy arrays, so each .sel() slice is
        # self-contained even after the source handle is gone.
        self._ds = xr.open_dataset(Path(nc_path)).load()
        self._period_ids = [
            pd.Timestamp(t).strftime("%Y-%m-%d") for t in self._ds["time"].values
        ]

    async def periods(self, start: str, end: str) -> list[str]:
        # This forecast's periods are exactly whatever the source file
        # contains — there's no lead-time arithmetic to do, just clip the
        # file's own period list to the requested [start, end] window. The
        # framework takes care of skipping periods already in the store.
        start_date, end_date = start[:10], end[:10]
        return [p for p in self._period_ids if start_date <= p <= end_date]

    def fetch_period(self, period_id: str, bbox: list[float], **params) -> xr.Dataset:
        da = self._ds[self.source_variable].sel(time=period_id)
        da = da.rio.write_crs(self.crs)
        return normalize_period(da, variable=self.variable, period=period_id, bbox=bbox)