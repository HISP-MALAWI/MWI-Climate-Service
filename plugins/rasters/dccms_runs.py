"""One DCCMS forecast run as a forecast cube: (reference_time, lead_time, y, x).

The DCCMS grid APIs (grid/forecast, grid/hazards) serve the current forecast one lead day at a
time. Stored by the date each lead day predicts, a daily sync would keep the values already
stored for those dates and add only the newest day, so the stored forecast for tomorrow would
go stale while it looks current. OCS refuses a schedule for such a dataset.

Stored as one run per issue date instead, a sync appends the new run and keeps the earlier
ones, and the dataset can be scheduled. This is the forecast layout of OCS PR #339
(dhis2/open-climate-service, `shared/forecast.py`): `reference_time` is the issue date,
`lead_time` the days ahead, and `forecast_valid_time` the date each value describes. Maps,
aggregations and DHIS2 exports read the latest run as an ordinary daily series.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import xarray as xr

from open_climate_service.shared import forecast
from open_climate_service.streaming import normalize_period


def forecast_run(grids: list[xr.DataArray], leads: list[int], issue: date, *, variable: str) -> xr.Dataset:
    """Stack one run's lead-day grids, each `(y, x)`, into a `(reference_time=1, lead_time, y, x)` cube.

    The grids of one run share their grid; `join="override"` keeps the first grid's
    coordinates, so a lead day whose derived axis differs in the last decimal does not
    misalign the stack.
    """
    if len(grids) != len(leads) or not grids:
        raise ValueError(f"A run needs one grid per lead day, got {len(grids)} grids for leads {leads}")
    stacked = xr.concat(grids, dim=forecast.LEAD_DIM, join="override", combine_attrs="override")
    stacked = stacked.assign_coords({forecast.LEAD_DIM: np.asarray(leads, dtype="int32")})
    return forecast.as_run(normalize_period(stacked, variable=variable), np.datetime64(issue, "ns"), "day")
