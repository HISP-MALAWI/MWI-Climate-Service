# Malawi Climate Service

This project is based on the [Open Climate Service](https://dhis2.github.io/climate-api/) and provides climate data for Sri Lanka, including ERA5-Land reanalysis, CHIRPS3 precipitation, and WorldPop population data.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/HISP-MALAWI/MWI-Climate-Service.git
   cd MWI-climate-service
   ```

2. **Install dependencies:**
   ```sh
   make install
   ```

3. **Configure environment:**
   ```sh
   cp .env.example .env
   # Edit .env and set CLIMATE_API_CONFIG to the absolute path of climate-api.yaml
   # Set PROJ_DATA to the rasterio proj_data path in your venv
   ```

## Usage

```sh
make run
```

The API starts at http://localhost:8002. Visit `/manage` to ingest datasets.

## Available Datasets

| Dataset | Period | Source |
|---|---|---|
| 2m temperature (ERA5-Land) | Hourly | DestinE Earth Data Hub |
| Total precipitation (ERA5-Land) | Hourly | DestinE Earth Data Hub |
| Total precipitation (CHIRPS3) | Daily | CHC UCSB |
| Total population (WorldPop) | Yearly | WorldPop |
| Elevation (Copernicus DEM GLO-30) | Static | AWS Open Data |
| Land Cover (ESA CCI) | Yearly | Copernicus CDS |
| Gross Primary Production (CLMS) | Dekadal | Copernicus CLMS |
| Vegetation Index NDVI (CLMS) | Dekadal | Copernicus CLMS |
| PM2.5 Surface Concentration (CAMS EAC4) | 3-hourly | Copernicus ADS |

## Population exposed to a hazard

The `population_exposure_by_hazard` workflow overlays a published categorical
hazard forecast with the built-in WorldPop Global2 100 m total-population
dataset. Set `hazard_categories` to the exposed integer codes in the selected
hazard collection and `no_hazard_value` to its unexposed code (defaults are
`[1, 2, 3, 4]` and `0`). Values outside those codes are treated as missing.
Hazard categories are resampled to the WorldPop grid using nearest-neighbour
resampling. The default heat-stress output remains available for backwards
compatibility; a generic four-dimensional managed Zarr cube can publish the
population separately for each configured category:

| `variable` | `output_dataset_id` | Contents |
| --- | --- | --- |
| `heatstress_exposure` (default) | `heatstress_population_exposure_forecast_daily` | Population per cell at each heat-stress exposure level (categories 1-3) |
| `population_by_hazard_category` | `dry_spell_population_by_hazard_category` | Population per cell at each configured hazard category |
| `exposed_population` | `dry_spell_population_exposure` | People per WorldPop cell where the hazard is 1-4 |
| `hazard_severity` | `dry_spell_hazard_severity_100m` | Hazard category 0-4 on the WorldPop grid |

`output_dataset_id` must name a template in `plugins/rasters/pop_exposure.yaml`
whose `variable` matches, so run the workflow once per variable.
The category cube's `exposure_level` dimension makes it possible to select
population exposure for a specific configured category. Each level contains
the population in cells with exactly that hazard category; unexposed cells
contain zero. It does not include a separate hazard-level band. For example,
to use hazard classes 2, 4, and 8 with -1 as the unexposed code, set
`hazard_categories` to `[2, 4, 8]` and `no_hazard_value` to `-1`.

`hazard_temporal_extent` can span multiple forecast dates; use one WorldPop
year for `population_temporal_extent`, a bounding box, and
`heatstress_hazard_forecast_daily` as the `hazard_dataset_id` (the default).
For other hazards, select the matching output template and `variable`; add a
template in `plugins/rasters/pop_exposure.yaml` for each published variable
and hazard.
Zarr output cannot be produced synchronously, so submit it as a batch job
(`POST /jobs`, then `POST /jobs/{id}/results`):

```json
{
  "process": {
    "process_graph": {
      "exposure": {
        "process_id": "population_exposure_by_hazard_workflow",
        "arguments": {
          "hazard_dataset_id": "dry_spell_hazard",
          "hazard_temporal_extent": ["2026-01-01", "2026-01-01"],
          "population_temporal_extent": ["2025-01-01", "2025-12-31"],
          "hazard_categories": [1, 2, 3, 4],
          "no_hazard_value": 0,
          "spatial_extent": {
            "west": 32.675,
            "east": 35.916666,
            "south": -17.133333,
            "north": -9.366666
          },
          "variable": "exposed_population",
          "output_dataset_id": "dry_spell_population_exposure"
        },
        "result": true
      }
    }
  }
}
```

Both source collections must already be ingested for the requested area and
period. The output dataset is registered and published by `save_result`.
