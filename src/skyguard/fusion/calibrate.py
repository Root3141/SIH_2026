"""Calibrate and persist the frozen SkyGuard detector state.

This is the one-time offline calibration step for the prototype. It reads
only the clean historical 2023-2025 dataset:

* Statistical calibration learns hourly baselines and rate-of-change
  thresholds.
* Spatial calibration learns station neighbors and residual thresholds.
* The existing frozen LSTM calibration is copied without retraining or
  modification.

The injected 2026 demo dataset is intentionally not read by this module.
Synthetic labels belong to evaluation and demo annotation, never detector
calibration.

Run from the project root with::

    python -m skyguard.fusion.calibrate

Artifacts are written to ``artifacts/skyguard_v1`` by default.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from skyguard.config.paths import (
    HISTORICAL_CSV,
    HISTORICAL_PARQUET,
    PROJECT_ROOT,
)
from skyguard.detectors.spatial import (
    SpatialConfig,
    calibrate_spatial_detector,
)
from skyguard.detectors.statistical import (
    StatisticalConfig,
    calibrate_statistical_detector,
)

VARIABLES = ["temperature", "humidity", "pressure"]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "skyguard_v1"
STATISTICAL_ARTIFACT = ARTIFACTS_DIR / "statistical.pkl"
SPATIAL_ARTIFACT = ARTIFACTS_DIR / "spatial.pkl"
LSTM_ARTIFACT = ARTIFACTS_DIR / "lstm.pkl"
FUSION_CONFIG = ARTIFACTS_DIR / "fusion_config.json"
LSTM_CALIBRATION_CACHE = (
    PROJECT_ROOT / "results" / "lstm_autoencoder" / "calibration.pkl"
)
REQUIRED_COLUMNS = {
    "station_id",
    "timestamp",
    "latitude",
    "longitude",
    *VARIABLES,
}


def load_historical_dataset(
    parquet_path: Path = HISTORICAL_PARQUET,
    csv_path: Path = HISTORICAL_CSV,
) -> tuple[pd.DataFrame, Path]:
    """Load and validate the clean historical calibration dataset."""

    if parquet_path.exists():
        dataframe = pd.read_parquet(parquet_path)
        source_path = parquet_path
    elif csv_path.exists():
        dataframe = pd.read_csv(csv_path)
        source_path = csv_path
    else:
        raise FileNotFoundError(
            "Historical calibration dataset not found. Checked:\n"
            f"  {parquet_path}\n"
            f"  {csv_path}"
        )

    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"Historical dataset {source_path} is missing required columns: {missing}"
        )

    dataframe = dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    if dataframe["timestamp"].isna().any():
        raise ValueError(
            f"Historical dataset {source_path} contains invalid timestamps."
        )

    years = dataframe["timestamp"].dt.year
    if not years.between(2023, 2025).all():
        raise ValueError(
            f"Historical dataset {source_path} contains timestamps outside 2023-2025."
        )

    return (
        dataframe.sort_values(["station_id", "timestamp"]).reset_index(drop=True),
        source_path,
    )


def _station_coordinates(dataframe: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Build one validated coordinate pair for each historical station."""

    coordinates = dataframe[["station_id", "latitude", "longitude"]].drop_duplicates()
    if coordinates["station_id"].duplicated().any():
        raise ValueError(
            "Historical station coordinates are inconsistent within a station."
        )

    return {
        str(row.station_id): (float(row.latitude), float(row.longitude))
        for row in coordinates.itertuples(index=False)
    }


def _save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calibrate_and_save(
    historical_path: Path = HISTORICAL_PARQUET,
    lstm_cache: Path = LSTM_CALIBRATION_CACHE,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Path]:
    """Calibrate from historical data and save all runtime artifacts."""

    historical, source_path = load_historical_dataset(
        parquet_path=historical_path,
        csv_path=HISTORICAL_CSV,
    )
    if not lstm_cache.exists():
        raise FileNotFoundError(
            "Frozen LSTM calibration cache not found:\n"
            f"  {lstm_cache}\n"
            "Run the existing LSTM calibration workflow before Step 2."
        )

    statistical_config = StatisticalConfig()
    spatial_config = SpatialConfig()
    statistical_calibration = calibrate_statistical_detector(
        historical,
        VARIABLES,
        statistical_config,
    )
    spatial_calibration = calibrate_spatial_detector(
        historical,
        VARIABLES,
        _station_coordinates(historical),
        spatial_config,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    statistical_path = artifacts_dir / STATISTICAL_ARTIFACT.name
    spatial_path = artifacts_dir / SPATIAL_ARTIFACT.name
    lstm_path = artifacts_dir / LSTM_ARTIFACT.name
    config_path = artifacts_dir / FUSION_CONFIG.name

    _save_pickle(
        statistical_path,
        {
            "detector": "statistical",
            "variables": VARIABLES,
            "config": statistical_config,
            "calibration": statistical_calibration,
        },
    )
    _save_pickle(
        spatial_path,
        {
            "detector": "spatial",
            "variables": VARIABLES,
            "config": spatial_config,
            "calibration": spatial_calibration,
        },
    )
    shutil.copyfile(lstm_cache, lstm_path)

    manifest = {
        "artifact_version": "skyguard_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_source": str(source_path),
        "calibration_period": {
            "start": historical["timestamp"].min().isoformat(),
            "end": historical["timestamp"].max().isoformat(),
        },
        "variables": VARIABLES,
        "stations": int(historical["station_id"].nunique()),
        "rows": int(len(historical)),
        "detectors": {
            "statistical": statistical_path.name,
            "spatial": spatial_path.name,
            "lstm": lstm_path.name,
        },
        "lstm_source": str(lstm_cache),
        "fusion_policy": {
            "name": "any_two_of_three",
            "detectors": ["statistical", "spatial", "lstm_ae"],
            "minimum_votes": 2,
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print("\nSKYGUARD DETECTOR CALIBRATION")
    print(f"Source:                 {source_path}")
    print(f"Rows:                   {len(historical):,}")
    print(f"Stations:               {historical['station_id'].nunique():,}")
    print(
        f"Period:                 {historical['timestamp'].min()} to "
        f"{historical['timestamp'].max()}"
    )
    print("Synthetic 2026 data:    NOT READ")
    print(f"Statistical artifact:   {statistical_path}")
    print(f"Spatial artifact:       {spatial_path}")
    print(f"Frozen LSTM artifact:   {lstm_path}")
    print(f"Fusion configuration:   {config_path}")

    return {
        "statistical": statistical_path,
        "spatial": spatial_path,
        "lstm": lstm_path,
        "fusion_config": config_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse optional source, cache, and artifact paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=HISTORICAL_PARQUET)
    parser.add_argument("--lstm-cache", type=Path, default=LSTM_CALIBRATION_CACHE)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Run historical-only calibration from the command line."""

    args = parse_args()
    calibrate_and_save(args.historical, args.lstm_cache, args.artifacts_dir)


if __name__ == "__main__":
    main()
