"""
schema.py

Column registries and split season definitions for every data source.

Keeping these here rather than inside dataset classes means the joint
dataset, trainers, loss masks, and eval code can import column lists
without importing the dataset classes themselves.

Each source uses a flat set of prefixed module-level variables so that
imports are unambiguous when multiple schemas are in scope:

    from datasets.schema import IBTRACS_PRIMARY_TARGET_COLS, IBTRACS_TRAIN_SEASONS
"""


# ---------------------------------------------------------------------------
# IBTrACS — North Atlantic / Caribbean-Gulf best tracks
# ---------------------------------------------------------------------------

IBTRACS_META_COLS: list[str] = [
    "SID",           # str,             storm identifier e.g. 2017242N16333
    "NAME",          # str,             storm name
    "SEASON",        # float32,         season year
    "BASIN",         # str,             ocean basin code e.g. NA
    "SUBBASIN",      # str,             sub-basin code
    "ISO_TIME",      # int64,            Unix nanoseconds, convert via pd.to_datetime
    "LAT",           # float32,         storm center latitude degrees
    "LON",           # float32,         storm center longitude degrees
    "TRACK_TYPE",    # str,             track type
    "IFLAG",         # str,             interpolation flag
    "USA_AGENCY",    # str,             source agency
    "USA_ATCF_ID",   # str,             ATCF storm identifier
    "USA_RECORD",    # str,             record identifier
    "USA_STATUS",    # str,             storm status e.g. HU TS TD EX
    "USA_SSHS",      # float32,         Saffir-Simpson category -3 to 5
]

IBTRACS_PRIMARY_TARGET_COLS: list[str] = [
    "USA_WIND",      # float32, m/s,    max sustained 1-min wind speed
    "USA_PRES",      # float32, Pa,     minimum central pressure
    "USA_POCI",      # float32, Pa,     pressure of outermost closed isobar
    "USA_RMW",       # float32, m,      radius of maximum winds
    "STORM_SPEED",   # float32, m/s,    storm translation speed
    "STORM_DIR",     # float32, deg CW from north, storm translation direction
]

IBTRACS_SECONDARY_TARGET_COLS: list[str] = [
    "USA_R17MS_NE", "USA_R17MS_SE", "USA_R17MS_SW", "USA_R17MS_NW",  # float32, m, 34 kt wind radius per quadrant
    "USA_R26MS_NE", "USA_R26MS_SE", "USA_R26MS_SW", "USA_R26MS_NW",  # float32, m, 50 kt wind radius per quadrant
    "USA_R33MS_NE", "USA_R33MS_SE", "USA_R33MS_SW", "USA_R33MS_NW",  # float32, m, 64 kt wind radius per quadrant
    "USA_ROCI",      # float32, m,      radius of outermost closed isobar
    "USA_EYE",       # float32, m,      eye diameter
    "USA_SEAHGT",    # float32, m,      sea height
    "USA_SEARAD_NE", "USA_SEARAD_SE", "USA_SEARAD_SW", "USA_SEARAD_NW",  # float32, m, sea radii per quadrant
]

IBTRACS_ALL_TARGET_COLS: list[str] = (
    IBTRACS_PRIMARY_TARGET_COLS + IBTRACS_SECONDARY_TARGET_COLS
)

# Temporal splits — season-based, no row-level randomisation.
# Hard test set (multi-storm timesteps) is loaded separately from
# ibtracs_multi_storm_times.npz and withheld entirely from train and val.
IBTRACS_TRAIN_SEASONS: list[int] = list(range(2005, 2021))   # 2005–2020 inclusive
IBTRACS_VAL_SEASONS:   list[int] = [2021, 2022]
IBTRACS_TEST_SEASONS:  list[int] = list(range(2023, 2026))    # 2023–2025 inclusive
