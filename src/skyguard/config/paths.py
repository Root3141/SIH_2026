"""Project path configuration."""

from pathlib import Path


# parents:
# 0 = config
# 1 = skyguard
# 2 = src
# 3 = project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_STATISTICAL_DIR = RESULTS_DIR / "statistical"
RESULTS_FUSION_DIR = RESULTS_DIR / "fusion"
RESULTS_SIMULATION_DIR = RESULTS_DIR / "simulation"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

HISTORICAL_PARQUET = RAW_DATA_DIR / "ncr_weather_historical.parquet"
HISTORICAL_CSV = RAW_DATA_DIR / "ncr_weather_historical.csv"
PRESENT_PARQUET = RAW_DATA_DIR / "ncr_weather_2026_present.parquet"
PRESENT_CSV = RAW_DATA_DIR / "ncr_weather_2026_present.csv"