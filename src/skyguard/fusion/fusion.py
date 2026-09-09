"""Runtime SkyGuard fusion over the frozen detector artifacts.

The public interface is :func:`run_fusion`. It loads the saved statistical,
spatial, and LSTM calibration state, runs the existing detectors, and applies
the prototype's deterministic 2-of-3 policy. Synthetic ground-truth columns,
when present in the input dataframe, are retained as data but are never read
by this module when making detector or fusion decisions.

Run from the project root to prepare a small smoke-test output::

    python -m skyguard.fusion.fusion
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from skyguard.config.paths import PROJECT_ROOT
from skyguard.detectors.lstm_autoencoder import run_lstm_autoencoder_detector
from skyguard.detectors.spatial import run_spatial_detector
from skyguard.detectors.statistical import run_statistical_detector

VARIABLES = ["temperature", "humidity", "pressure"]
KEY_COLUMNS = ["station_id", "timestamp"]
REQUIRED_COLUMNS = set(KEY_COLUMNS) | set(VARIABLES)
DETECTOR_COLUMNS = [*KEY_COLUMNS, *VARIABLES]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "skyguard_v1"
STATISTICAL_ARTIFACT = ARTIFACTS_DIR / "statistical.pkl"
SPATIAL_ARTIFACT = ARTIFACTS_DIR / "spatial.pkl"
LSTM_ARTIFACT = ARTIFACTS_DIR / "lstm.pkl"
MINIMUM_VOTES = 2
FINAL_SEVERITY_BY_VOTES = {
    0: "normal",
    1: "suspicious",
    2: "anomaly",
    3: "critical",
}


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found: {path}\n"
            "Run `python -m skyguard.fusion.calibrate` first."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_artifacts(artifacts_dir: Path) -> tuple[dict, dict, object]:
    """Load and validate the three detector artifacts."""

    statistical = _load_pickle(artifacts_dir / STATISTICAL_ARTIFACT.name)
    spatial = _load_pickle(artifacts_dir / SPATIAL_ARTIFACT.name)
    lstm = _load_pickle(artifacts_dir / LSTM_ARTIFACT.name)

    if statistical.get("detector") != "statistical":
        raise ValueError("Invalid statistical calibration artifact.")
    if spatial.get("detector") != "spatial":
        raise ValueError("Invalid spatial calibration artifact.")

    for artifact, name in ((statistical, "statistical"), (spatial, "spatial")):
        if artifact.get("variables") != VARIABLES:
            raise ValueError(
                f"{name} calibration variables do not match runtime variables: "
                f"{artifact.get('variables')}"
            )

    if not hasattr(lstm, "variables") or set(lstm.variables) != set(VARIABLES):
        raise ValueError("Invalid or incompatible frozen LSTM calibration artifact.")

    return statistical, spatial, lstm


def _validate_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(f"Fusion input is missing required columns: {missing}")

    result = dataframe.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    if result["timestamp"].isna().any():
        raise ValueError("Fusion input contains invalid timestamps.")

    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _weather_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep detector input independent from evaluation metadata and labels."""

    return dataframe[DETECTOR_COLUMNS].copy()


def _join_detector_columns(
    base: pd.DataFrame,
    detector_result: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    selected = detector_result[KEY_COLUMNS + columns].copy()
    if selected.duplicated(KEY_COLUMNS).any():
        raise ValueError("Detector output contains duplicate station/timestamp keys.")

    return base.merge(selected, on=KEY_COLUMNS, how="left", validate="one_to_one")


def _apply_two_of_three(result: pd.DataFrame) -> pd.DataFrame:
    alert_columns = [
        "statistical_alert",
        "spatial_alert",
        "lstm_ae_alert",
    ]
    result = result.copy()
    for column in alert_columns:
        result[column] = result[column].fillna(False).astype(bool)

    result["detector_votes"] = result[alert_columns].astype(int).sum(axis=1)
    result["final_alert"] = result["detector_votes"] >= MINIMUM_VOTES
    result["final_severity"] = assign_final_severity(result["detector_votes"])
    result["final_confidence"] = result["detector_votes"] / len(alert_columns)
    return result


def assign_final_severity(detector_votes: pd.Series) -> pd.Series:
    """Map detector agreement to the prototype's four severity tiers.

    The labels intentionally follow the existing detector and fusion package
    convention: ``normal``, ``suspicious``, ``anomaly``, and ``critical``.
    Detector votes are expected to be integers from zero through three.
    """

    invalid = ~detector_votes.isin(FINAL_SEVERITY_BY_VOTES)
    if invalid.any():
        raise ValueError("Detector votes must be integers in the range 0-3.")
    return detector_votes.map(FINAL_SEVERITY_BY_VOTES)


def run_fusion(
    dataframe: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Run the frozen detectors and return their fused runtime results.

    The returned dataframe preserves the input weather and metadata columns,
    adds detector-prefixed outputs, and applies an alert when at least two of
    the three detector alert columns are true. Detector input is restricted to
    the required weather fields, so evaluation labels cannot influence the
    runtime decision.
    """

    base = _validate_input(dataframe)
    detector_input = _weather_input(base)
    statistical, spatial, lstm = _load_artifacts(artifacts_dir)

    statistical_result = run_statistical_detector(
        detector_input,
        VARIABLES,
        statistical["calibration"],
        statistical["config"],
    )
    spatial_result = run_spatial_detector(
        detector_input,
        VARIABLES,
        spatial["calibration"],
        spatial["config"],
    )
    lstm_result = run_lstm_autoencoder_detector(
        detector_input,
        lstm,
        lstm.config,
    )

    result = base
    result = _join_detector_columns(
        result,
        statistical_result,
        [
            column
            for column in statistical_result.columns
            if column.startswith("statistical_")
        ],
    )
    result = _join_detector_columns(
        result,
        spatial_result,
        [
            column
            for column in spatial_result.columns
            if column.startswith("spatial_") or "_spatial_" in column
        ],
    )
    result = _join_detector_columns(
        result,
        lstm_result,
        [column for column in lstm_result.columns if column.startswith("lstm_ae_")],
    )

    return _apply_two_of_three(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic" / "skyguard_demo_2026.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "prototype"
        / "skyguard_demo_2026_results.parquet",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Fusion input not found: {args.input}")

    result = run_fusion(pd.read_parquet(args.input), args.artifacts_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print("\nSKYGUARD 2-OF-3 FUSION")
    print(f"Input rows:             {len(result):,}")
    print(f"Final alerts:           {int(result['final_alert'].sum()):,}")
    print(f"Output:                 {args.output}")


if __name__ == "__main__":
    main()
