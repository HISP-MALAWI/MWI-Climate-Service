"""Aggregate a climate dataset to DHIS2 organisation units and import it into DHIS2.

End-to-end pipeline driven from Python:

  1. Fetch organisation unit boundaries from DHIS2 as GeoJSON.
  2. Run the built-in ``aggregate_to_dhis2_json`` workflow on an Open Climate Service instance 
      — it loads a dataset, aggregates it within each org-unit polygon, and returns a ready-to-import DHIS2 ``dataValueSet``.
  3. Import that ``dataValueSet`` back into DHIS2.

The workflow uses each org unit's GeoJSON ``id`` (its DHIS2 UID) as the ``orgUnit``, so the result imports into DHIS2 without any remapping.

Requires:
  pip install open-climate-service "dhis2-client @ git+https://github.com/dhis2/dhis2-python-client.git"
  A running Open Climate Service instance with the dataset published.
  A DHIS2 instance whose org units have geometry, plus a target data element.

Adjust the configuration constants below for your instances.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from dhis2_client import DHIS2Client
from dhis2_client.settings import ClientSettings

from open_climate_service import ClimateService

# --- Open Climate Service ---------------------------------------------------
load_dotenv()
OCS_BASE_URL = os.environ.get("OCS_BASE_URL", "http://127.0.0.1:8002")
DATASET_ID = os.environ.get("DATASET_ID", "")  

# a published collection (see /datasets)
TEMPORAL_EXTENT = ["2020-06-30", "2026-03-31"]
METHOD = "mean"  # mean (default), min, max, or sum
PERIOD_TYPE = "month"

# --- DHIS2 ------------------------------------------------------------------
DHIS2_BASE_URL = os.environ.get("DHIS2_BASE_URL", "")
DHIS2_USERNAME = os.environ.get("DHIS2_USERNAME", "")
DHIS2_PASSWORD = os.environ.get("DHIS2_PASSWORD", "")
ORG_UNIT_LEVEL =  os.environ.get("ORG_UNIT_DISTRICT_LEVEL", None) # org unit level to aggregate to
DATA_ELEMENT_ID = os.environ.get("TEST_DATA_ELEMENT_ID", "")  # DHIS2 data element to import the values into


def main() -> None:
    """Fetch org units from DHIS2, aggregate them on Open Climate Service, import back."""
    dhis2 = DHIS2Client(
        settings=ClientSettings(
            base_url=DHIS2_BASE_URL,
            username=DHIS2_USERNAME,
            password=DHIS2_PASSWORD,
        )
    )

    # 1. Fetch org unit boundaries from DHIS2 as GeoJSON.
    #    Each feature's `id` is the DHIS2 org unit UID, which the workflow uses as the orgUnit.
    org_units = dhis2.get_org_units_geojson(level=ORG_UNIT_LEVEL)
    print(f"Fetched {len(org_units['features'])} org units from DHIS2 (level {ORG_UNIT_LEVEL})")

    # 2. Run the aggregation workflow on Open Climate Service -> DHIS2 dataValueSet.
    service = ClimateService(OCS_BASE_URL)
    data_value_set = service.execute(
        {
            "agg": {
                "process_id": "aggregate_to_dhis2_json",
                "arguments": {
                    "dataset_id": DATASET_ID,
                    "temporal_extent": TEMPORAL_EXTENT,
                    "geometries": org_units,
                    "data_element_id": DATA_ELEMENT_ID,
                    "method": METHOD,
                    "period_type": PERIOD_TYPE,
                },
                "result": True,
            }
        }
    )
    values = data_value_set.get("dataValues", [])
    print(f"Workflow produced {len(values)} data values for data element {DATA_ELEMENT_ID}")
    if not values:
        print("No data values produced — check the dataset id, temporal extent, and org unit geometries.")
        return
    print(f"  e.g. {values[0]}")

    # 3. Import the dataValueSet into DHIS2.
    report = dhis2.post_data_value_set(data_value_set)
    import_count = report.get("response", {}).get("importCount", report)
    print("DHIS2 import summary:", import_count)


if __name__ == "__main__":
    main()
