"""Heat-stress hazard levels derived from daily mean temperature forecasts."""

from __future__ import annotations

from typing import Any

import xarray as xr

from open_climate_service.streaming import normalize_period

from .daily_forecast import DailyForecastPlugin, _DEFAULT_MAX_FORECAST_DAYS


class HeatStressHazardPlugin(DailyForecastPlugin):
    """Classify DCCMS daily mean temperatures into heat-stress levels."""

    def __init__(
        self,
        base_url: str,
        dataset: str = "tmean",
        hazard_variable: str = "heatstress_hazard",
        max_forecast_days: int = _DEFAULT_MAX_FORECAST_DAYS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            dataset=dataset,
            max_forecast_days=max_forecast_days,
            **kwargs,
        )
        self._hazard_variable = hazard_variable

    def fetch_period(self, period_id: str, bbox: list[float], **kwargs: Any) -> xr.Dataset:
        temperature = super().fetch_period(period_id, bbox, **kwargs)[self._variable]
        hazard = xr.where(
            temperature < 35,
            0,
            xr.where(temperature < 38, 1, xr.where(temperature <= 40, 2, 3)),
        ).where(temperature.notnull())
        hazard.attrs.update(
            units="1",
            long_name="Heat-stress hazard level from daily mean temperature",
            flag_values=[0, 1, 2, 3],
            flag_meanings="no_hazard level_1 level_2 level_3",
        )
        return normalize_period(
            hazard, variable=self._hazard_variable, period=period_id
        ).load()
