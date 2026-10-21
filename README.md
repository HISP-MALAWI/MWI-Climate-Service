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

## Experimental dataset plugins (ocs-plugins)

The instance also installs [ocs-plugins](https://github.com/dhis2-chap/ocs-plugins), experimental
dataset plugins for OCS, found through their entry point: no `plugins_dir` change is needed. Their
datasets are listed at `GET /data-sources` and on the Data sources page next to the built-in ones,
among them ERA5-Land daily statistics, CAMS dust, PM10 and PM2.5, GloFAS river discharge, IMERG,
MODIS NDVI, EVI and land surface temperature, CLMS GPP, NPP and NDVI, ESA CCI and WorldCover land
cover with their classes, and the 90 m DEM. Several need credentials; see the ocs-plugins README.
They are for testing, not yet for production.

A template in `plugins/` with the same id as one in ocs-plugins overrides it. That is the case for
`clms_gpp_dekadal`, `clms_ndvi_dekadal` and `modis_landcover_yearly`. Other local templates read
the same sources under other ids: `cams_pm25_hourly`, `esa_landcover_yearly`, `clms_npp_dekad`
and `copernicus_dem_elevation`. Where the ocs-plugins version does what is needed, the local
plugin and template can be removed.

OCS itself is pinned to the `feat/forecast-cubes` branch (OCS PR #339) for the forecast datasets,
in `[tool.uv.sources]` and in `override-dependencies`, since ocs-plugins pins OCS `main` in its
own sources. Both move back to `main` once #339 merges. `make upgrade` re-locks OCS and
ocs-plugins against their branch heads.

## Population exposed to a hazard

The `population_exposure_by_hazard` workflow overlays a published hazard
snapshot with the built-in WorldPop Global2 100 m total-population dataset.
Hazard categories 1-4 are treated as exposed (4 is the most severe); category
0 is unexposed. Severity is resampled to the WorldPop grid using
nearest-neighbour resampling. Each run publishes one variable as a managed
Zarr dataset, chosen with `variable`:

| `variable` | `output_dataset_id` | Contents |
| --- | --- | --- |
| `exposed_population` (default) | `dry_spell_population_exposure` | People per WorldPop cell where the hazard is 1-4 |
| `hazard_severity` | `dry_spell_hazard_severity_100m` | Hazard category 0-4 on the WorldPop grid |

`output_dataset_id` must name a template in `plugins/rasters/pop_exposure.yaml`
whose `variable` matches, so run the workflow once per variable.

Provide a single date for `hazard_temporal_extent`, a single WorldPop year for
`population_temporal_extent`, a bounding box and the `hazard_dataset_id` (for
example `dry_spell_hazard`). Zarr output cannot be produced synchronously, so
submit it as a batch job (`POST /jobs`, then `POST /jobs/{id}/results`):

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
