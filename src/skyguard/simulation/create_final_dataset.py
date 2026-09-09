"""Create the reproducible synthetic benchmark used by the SkyGuard demo.

This module is the Step 1 boundary in the prototype workflow. It reads the
clean 2026 observations, injects the fixed benchmark anomalies using the
project's existing injector, and writes two immutable-by-convention Parquet
artifacts:

* ``data/synthetic/skyguard_demo_2026.parquet`` - detector input plus
  evaluation metadata;
* ``data/synthetic/skyguard_demo_2026_log.parquet`` - one row per injected
  anomaly event.

The benchmark uses a 2% target anomaly rate and seed 42, matching the shared
benchmark used by the combined evaluator. The injector creates UUID event
identifiers for general evaluation use; this script replaces them with stable
sequential identifiers so repeated runs produce the same values and event
log. Ground-truth columns are retained for offline evaluation only and must
not be used by detector inference.

Run from the project root with::

    python -m skyguard.simulation.create_final_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from skyguard.config.paths import (
    PRESENT_CSV,
    PRESENT_PARQUET,
    SYNTHETIC_DATA_DIR,
)
from skyguard.simulation.anomaly_injector import (
    AnomalyConfig,
    inject_anomalies,
)

VARIABLES = ["temperature", "humidity", "pressure"]
ANOMALY_RATE = 0.02
RANDOM_SEED = 42
REQUIRED_COLUMNS = {"station_id", "timestamp", *VARIABLES}
OUTPUT_DATASET = SYNTHETIC_DATA_DIR / "skyguard_demo_2026.parquet"
OUTPUT_LOG = SYNTHETIC_DATA_DIR / "skyguard_demo_2026_log.parquet"


def load_present_dataset(
    parquet_path: Path = PRESENT_PARQUET,
    csv_path: Path = PRESENT_CSV,
) -> pd.DataFrame:
    """Load and validate the clean 2026 source dataset.

    Parquet is preferred because it is the repository's normal evaluator
    format. CSV is supported as a documented fallback for fresh checkouts or
    data refreshes that have not yet produced the Parquet companion file.
    """

    if parquet_path.exists():
        dataframe = pd.read_parquet(parquet_path)
        source = parquet_path
    elif csv_path.exists():
        dataframe = pd.read_csv(csv_path)
        source = csv_path
    else:
        raise FileNotFoundError(
            "2026 source dataset not found. Checked:\n"
            f"  {parquet_path}\n"
            f"  {csv_path}"
        )

    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"2026 source dataset {source} is missing required columns: {missing}"
        )

    dataframe = dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    if dataframe["timestamp"].isna().any():
        raise ValueError(f"2026 source dataset {source} contains invalid timestamps.")

    if not (dataframe["timestamp"].dt.year == 2026).all():
        raise ValueError(
            f"2026 source dataset {source} contains timestamps outside 2026."
        )

    return dataframe.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def _stable_event_ids(
    contaminated: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace injector UUIDs with deterministic IDs in both output tables."""

    contaminated = contaminated.copy()
    events = events.copy()
    replacements: dict[str, str] = {}

    for event_number, event_id in enumerate(events["event_id"].astype(str), start=1):
        replacements[event_id] = f"event-{event_number:06d}"

    if replacements:
        events["event_id"] = events["event_id"].astype(str).replace(replacements)

        for column in (
            "synthetic_anomaly_event_id",
            "anomaly_id",
        ):
            if column not in contaminated:
                continue

            def replace_ids(value: object) -> object:
                if pd.isna(value):
                    return value
                return "|".join(
                    replacements.get(part, part) for part in str(value).split("|")
                )

            contaminated[column] = contaminated[column].map(replace_ids)

    return contaminated, events


def _add_evaluation_aliases(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical ground-truth names used by existing evaluators."""

    dataframe = dataframe.copy()
    aliases = {
        "is_anomaly": "synthetic_anomaly",
        "anomaly_id": "synthetic_anomaly_event_id",
        "anomaly_type": "synthetic_anomaly_type",
        "anomaly_variable": "synthetic_anomaly_variable",
        "anomaly_severity": "synthetic_anomaly_severity",
    }
    for alias, source in aliases.items():
        dataframe[alias] = dataframe[source]

    dataframe["is_anomaly"] = dataframe["is_anomaly"].fillna(False).astype(bool)
    return dataframe


def create_final_dataset(
    source_path: Path = PRESENT_PARQUET,
    output_dataset: Path = OUTPUT_DATASET,
    output_log: Path = OUTPUT_LOG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and save the fixed 2026 demo dataset and injection log."""

    clean_dataframe = load_present_dataset(
        parquet_path=source_path, csv_path=PRESENT_CSV
    )
    config = AnomalyConfig(
        anomaly_rate=ANOMALY_RATE,
        random_seed=RANDOM_SEED,
    )
    contaminated, events = inject_anomalies(
        clean_dataframe,
        VARIABLES,
        config,
    )
    contaminated, events = _stable_event_ids(contaminated, events)
    contaminated = _add_evaluation_aliases(contaminated)
    contaminated = contaminated.sort_values(["station_id", "timestamp"]).reset_index(
        drop=True
    )

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    contaminated.to_parquet(output_dataset, index=False)
    events.to_parquet(output_log, index=False)

    anomaly_count = int(contaminated["is_anomaly"].sum())
    print("\nSKYGUARD FINAL 2026 DEMO DATASET")
    print(f"Source:                 {source_path}")
    print(f"Rows:                   {len(contaminated):,}")
    print(f"Stations:               {contaminated['station_id'].nunique():,}")
    print(
        f"Period:                 {contaminated['timestamp'].min()} to {contaminated['timestamp'].max()}"
    )
    print(f"Random seed:            {RANDOM_SEED}")
    print(f"Target anomaly rate:    {ANOMALY_RATE:.2%}")
    print(
        f"Anomalous observations: {anomaly_count:,} ({anomaly_count / len(contaminated):.2%})"
    )
    print(f"Anomaly events:         {len(events):,}")
    print(f"Dataset output:         {output_dataset}")
    print(f"Event log output:       {output_log}")

    return contaminated, events


def parse_args() -> argparse.Namespace:
    """Parse optional paths for reproducibility checks and data refreshes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PRESENT_PARQUET)
    parser.add_argument("--output", type=Path, default=OUTPUT_DATASET)
    parser.add_argument("--log-output", type=Path, default=OUTPUT_LOG)
    return parser.parse_args()


def main() -> None:
    """Generate the final benchmark from the command line."""

    args = parse_args()
    create_final_dataset(args.source, args.output, args.log_output)


if __name__ == "__main__":
    main()
