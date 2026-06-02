"""
schema.py

Column registries and split season definitions for every data source.

Keeping these here rather than inside dataset classes means the joint
dataset, trainers, loss masks, and eval code can import column lists
without importing the dataset classes themselves.

Each schema is a plain class with class-level lists — no instances needed.
New source schemas are added here when their data cleaning pipeline is
complete.
"""


# ---------------------------------------------------------------------------
# Season split definitions (shared across all source datasets)
# ---------------------------------------------------------------------------

TRAIN_SEASONS: list[int] = list(range(2005, 2021))   # 2005-2020 inclusive
VAL_SEASONS:   list[int] = [2021, 2022]
TEST_SEASONS:  list[int] = list(range(2023, 2026))    # 2023-2025 inclusive


# ---------------------------------------------------------------------------
# IBTrACS
# ---------------------------------------------------------------------------

class IBTrACSSchema:
    """Column registry for the cleaned IBTrACS North Atlantic npz files."""

    METADATA: list[str] = [
        "SID", "NAME", "SEASON", "BASIN", "SUBBASIN",
        "ISO_TIME", "LAT", "LON",
        "TRACK_TYPE", "IFLAG", "USA_AGENCY", "USA_ATCF_ID",
        "USA_RECORD", "USA_STATUS", "USA_SSHS",
    ]

    # Core intensity and structure parameters needed by parametric wind
    # field models (Holland 2008, CLIMADA TropCyclone).
    PRIMARY_TARGETS: list[str] = [
        "USA_WIND",        # max sustained 1-min wind speed (m/s)
        "USA_PRES",        # minimum central pressure (Pa)
        "USA_POCI",        # pressure of outermost closed isobar (Pa)
        "USA_RMW",         # radius of maximum winds (m)
        "STORM_SPEED",     # storm translation speed (m/s)
        "STORM_DIR",       # storm translation direction (degrees CW from N)
    ]

    # Quadrant wind radii and additional structural parameters.
    # Zero-imputed where wind speed is below the threshold.
    SECONDARY_TARGETS: list[str] = [
        "USA_R17MS_NE", "USA_R17MS_SE", "USA_R17MS_SW", "USA_R17MS_NW",  # 34 kt radii (m)
        "USA_R26MS_NE", "USA_R26MS_SE", "USA_R26MS_SW", "USA_R26MS_NW",  # 50 kt radii (m)
        "USA_R33MS_NE", "USA_R33MS_SE", "USA_R33MS_SW", "USA_R33MS_NW",  # 64 kt radii (m)
        "USA_ROCI",        # radius of outermost closed isobar (m)
        "USA_EYE",         # eye diameter (m)
        "USA_SEAHGT",      # sea height (m)
        "USA_SEARAD_NE", "USA_SEARAD_SE", "USA_SEARAD_SW", "USA_SEARAD_NW",
    ]

    ALL_TARGETS: list[str] = PRIMARY_TARGETS + SECONDARY_TARGETS
