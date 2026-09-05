"""
SkyGuard — Three-Detector Combined Evaluator
=============================================

Controlled baseline evaluation of:

    Statistical detector
    Spatial detector
    LSTM Autoencoder detector

The same synthetic anomaly benchmark is used for every detector.

This script is intentionally a baseline/diagnostic evaluator:
- no fusion-weight tuning
- no retraining
- no persistence
- no architecture changes
- one shared anomaly injection
- one LSTM inference pass
- LSTM alert thresholds simulated from the same cached scores

Compared LSTM operating points:
    1.00x calibration P99
    1.50x calibration P99
    1.75x calibration P99

Baseline:
    statistical + spatial only
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyguard.config.paths import (
    HISTORICAL_PARQUET,
    PRESENT_PARQUET,
    RESULTS_DIR,
    SYNTHETIC_DATA_DIR,
)
from skyguard.detectors.statistical import (
    StatisticalConfig,
    calibrate_statistical_detector,
    run_statistical_detector,
)
from skyguard.detectors.spatial import (
    SpatialConfig,
    calibrate_spatial_detector,
    run_spatial_detector,
)
from skyguard.detectors.lstm_autoencoder import (
    run_lstm_autoencoder_detector,
)
from skyguard.simulation.anomaly_injector import (
    AnomalyConfig,
    inject_anomalies,
)

VARIABLES = ["temperature", "humidity", "pressure"]

RANDOM_SEED = 42
ANOMALY_RATE = 0.02

KEY_COLUMNS = ["station_id", "timestamp"]

OUTPUT_DIR = RESULTS_DIR / "statistical_spatial_lstm"

LSTM_CALIBRATION_CACHE = RESULTS_DIR / "lstm_autoencoder" / "calibration.pkl"

INJECTED_PARQUET = (
    SYNTHETIC_DATA_DIR / "statistical_spatial_lstm_evaluation_injected.parquet"
)

LSTM_MULTIPLIERS = [1.00, 1.50, 1.75]
OVERLAP_BUFFER = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SkyGuard statistical + spatial + LSTM detectors."
    )

    parser.add_argument(
        "--lstm-multipliers",
        default="1.0,1.5,1.75",
        help=(
            "Comma-separated multipliers of calibration P99 to test. "
            "Default: 1.0,1.5,1.75"
        ),
    )

    return parser.parse_args()


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    if not HISTORICAL_PARQUET.exists():
        raise FileNotFoundError(f"Historical dataset not found:\n{HISTORICAL_PARQUET}")

    if not PRESENT_PARQUET.exists():
        raise FileNotFoundError(f"Evaluation dataset not found:\n{PRESENT_PARQUET}")

    historical_df = pd.read_parquet(HISTORICAL_PARQUET).copy()
    present_df = pd.read_parquet(PRESENT_PARQUET).copy()

    for df in (historical_df, present_df):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

    historical_df = historical_df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    present_df = present_df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    print("\nHistorical calibration data:")
    print(f"  Rows:     {len(historical_df):,}")
    print(f"  Stations: {historical_df['station_id'].nunique():,}")
    print(
        f"  Period:   "
        f"{historical_df['timestamp'].min()} "
        f"to "
        f"{historical_df['timestamp'].max()}"
    )

    print("\nUnseen evaluation data:")
    print(f"  Rows:     {len(present_df):,}")
    print(f"  Stations: {present_df['station_id'].nunique():,}")
    print(
        f"  Period:   "
        f"{present_df['timestamp'].min()} "
        f"to "
        f"{present_df['timestamp'].max()}"
    )

    return historical_df, present_df


def inject_shared_anomalies(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\n" + "=" * 80)
    print("INJECTING ONE SHARED SYNTHETIC BENCHMARK")
    print("=" * 80)

    config = AnomalyConfig(
        anomaly_rate=ANOMALY_RATE,
        random_seed=RANDOM_SEED,
    )

    corrupted_df, injection_log = inject_anomalies(
        clean_df.copy(),
        VARIABLES,
        config,
    )

    corrupted_df = corrupted_df.copy()

    if "is_anomaly" not in corrupted_df.columns:
        corrupted_df["is_anomaly"] = (
            corrupted_df.get(
                "synthetic_anomaly",
                pd.Series(False, index=corrupted_df.index),
            )
            .fillna(False)
            .astype(bool)
        )

    if "anomaly_id" not in corrupted_df.columns:
        corrupted_df["anomaly_id"] = corrupted_df.get("synthetic_anomaly_event_id")

    if "anomaly_type" not in corrupted_df.columns:
        corrupted_df["anomaly_type"] = corrupted_df.get("synthetic_anomaly_type")

    if "anomaly_variable" not in corrupted_df.columns:
        corrupted_df["anomaly_variable"] = corrupted_df.get(
            "synthetic_anomaly_variable"
        )

    if "anomaly_severity" not in corrupted_df.columns:
        corrupted_df["anomaly_severity"] = corrupted_df.get(
            "synthetic_anomaly_severity"
        )

    required = {
        "is_anomaly",
        "anomaly_id",
        "anomaly_type",
        "anomaly_variable",
        "anomaly_severity",
    }

    missing = sorted(required - set(corrupted_df.columns))

    if missing:
        raise ValueError(
            f"Missing ground-truth columns after anomaly injection: {missing}"
        )

    corrupted_df = corrupted_df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    corrupted_df.to_parquet(
        INJECTED_PARQUET,
        index=False,
    )

    anomaly_count = int(corrupted_df["is_anomaly"].fillna(False).sum())

    event_count = int(
        corrupted_df.loc[
            corrupted_df["is_anomaly"],
            "anomaly_id",
        ].nunique()
    )

    print(f"  Total observations:    {len(corrupted_df):,}")
    print(f"  Anomalous observations:{anomaly_count:,}")
    print(f"  Actual anomaly rate:   " f"{anomaly_count / len(corrupted_df):.3%}")
    print(f"  Anomaly events:        {event_count:,}")
    print(f"  Seed:                  {RANDOM_SEED}")
    print(f"  Saved benchmark:       {INJECTED_PARQUET}")

    return corrupted_df, injection_log


def calibrate_stat_spatial(
    historical_df: pd.DataFrame,
):
    print("\n" + "=" * 80)
    print("CALIBRATING STATISTICAL + SPATIAL DETECTORS")
    print("=" * 80)

    stat_config = StatisticalConfig()

    stat_calibration = calibrate_statistical_detector(
        historical_df,
        VARIABLES,
        stat_config,
    )

    station_coords = (
        historical_df[["station_id", "latitude", "longitude"]]
        .drop_duplicates()
        .set_index("station_id")
        .apply(
            lambda row: (
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )
        .to_dict()
    )

    spatial_config = SpatialConfig()

    spatial_calibration = calibrate_spatial_detector(
        historical_df,
        VARIABLES,
        station_coords,
        spatial_config,
    )

    print("\nStatistical configuration:")
    print(f"  ROC percentile:       {stat_config.roc_percentile}")
    print(f"  Z-score threshold:    {stat_config.zscore_threshold}")
    print(f"  Z-score windows:      {stat_config.zscore_windows}")
    print(f"  Persistence window:  {stat_config.persistence_window}")

    print("\nSpatial configuration:")
    print(f"  k neighbors:          {spatial_config.k_neighbors}")
    print(f"  IDW power:             {spatial_config.idw_power}")
    print(f"  Suspicious percentile:" f" {spatial_config.suspicious_percentile}")
    print(f"  Anomaly percentile:    " f"{spatial_config.anomaly_percentile}")

    return (
        stat_calibration,
        stat_config,
        spatial_calibration,
        spatial_config,
    )


def load_lstm_calibration():
    print("\n" + "=" * 80)
    print("LOADING FROZEN LSTM AUTOENCODER CALIBRATION")
    print("=" * 80)

    if not LSTM_CALIBRATION_CACHE.exists():
        raise FileNotFoundError(
            f"LSTM calibration cache not found:\n"
            f"{LSTM_CALIBRATION_CACHE}\n\n"
            "Run:\n"
            "  python src/skyguard/evaluation/"
            "evaluate_lstm_autoencoder.py --mode full"
        )

    with open(LSTM_CALIBRATION_CACHE, "rb") as f:
        calibration = pickle.load(f)

    config = calibration.config

    print(f"  Backend:              {calibration.backend}")
    print(f"  Stations calibrated:  {calibration.n_stations}")
    print(f"  Training windows:     {calibration.n_training_windows}")
    print(f"  Validation windows:   {calibration.n_validation_windows}")
    print(f"  Calibration P95:      {calibration.threshold_p95:.6f}")
    print(f"  Calibration P99:      {calibration.threshold_p99:.6f}")

    return calibration, config


def run_detectors(
    evaluation_df: pd.DataFrame,
    stat_calibration,
    stat_config,
    spatial_calibration,
    spatial_config,
    lstm_calibration,
    lstm_config,
) -> pd.DataFrame:

    print("\n" + "=" * 80)
    print("RUNNING THREE DETECTORS ON THE SAME DATA")
    print("=" * 80)

    stat_result = run_statistical_detector(
        evaluation_df.copy(),
        VARIABLES,
        stat_calibration,
        stat_config,
    )

    spatial_result = run_spatial_detector(
        evaluation_df.copy(),
        VARIABLES,
        spatial_calibration,
        spatial_config,
    )

    print("\nRunning LSTM Autoencoder...")
    lstm_result = run_lstm_autoencoder_detector(
        evaluation_df.copy(),
        lstm_calibration,
        lstm_config,
    )

    outputs = {
        "statistical": (stat_result.sort_values(KEY_COLUMNS).reset_index(drop=True)),
        "spatial": (spatial_result.sort_values(KEY_COLUMNS).reset_index(drop=True)),
        "lstm": (lstm_result.sort_values(KEY_COLUMNS).reset_index(drop=True)),
    }

    base = outputs["statistical"]

    for name, result in outputs.items():

        if len(base) != len(result):
            raise ValueError(
                f"Statistical and {name} outputs have different row counts."
            )

        if not base[KEY_COLUMNS].equals(result[KEY_COLUMNS]):
            raise ValueError(
                f"Statistical and {name} outputs "
                "do not share identical station/timestamp keys."
            )

    result = base.copy()

    spatial_columns = [
        column
        for column in outputs["spatial"].columns
        if (
            column
            in {
                "spatial_alert",
                "spatial_severity_level",
            }
            or column.endswith("_spatial_anomaly")
            or column.endswith("_spatial_suspicious")
            or "neighbor" in column.lower()
            or column.startswith("spatial_")
        )
    ]

    spatial_columns = [
        column for column in dict.fromkeys(spatial_columns) if column not in KEY_COLUMNS
    ]

    result = result.join(
        outputs["spatial"].set_index(KEY_COLUMNS)[spatial_columns],
        on=KEY_COLUMNS,
    )

    lstm_columns = [
        column
        for column in outputs["lstm"].columns
        if (
            column.startswith("lstm_ae_")
            or column.endswith("_lstm_ae_error")
            or column.endswith("_lstm_ae_expected")
            or column.endswith("_lstm_ae_residual")
        )
    ]

    lstm_columns = [
        column for column in dict.fromkeys(lstm_columns) if column not in KEY_COLUMNS
    ]

    result = result.join(
        outputs["lstm"].set_index(KEY_COLUMNS)[lstm_columns],
        on=KEY_COLUMNS,
    )

    required = [
        "statistical_alert",
        "spatial_alert",
        "lstm_ae_score",
    ]

    missing = [column for column in required if column not in result.columns]

    if missing:
        raise ValueError(f"Required detector outputs missing: {missing}")

    result["statistical_alert"] = result["statistical_alert"].fillna(False).astype(bool)

    result["spatial_alert"] = result["spatial_alert"].fillna(False).astype(bool)

    result["is_anomaly"] = result["is_anomaly"].fillna(False).astype(bool)

    result["lstm_ae_alert_original"] = (
        result["lstm_ae_alert"].fillna(False).astype(bool)
    )

    print("\nDetector output sizes:")
    print(f"  Rows: {len(result):,}")

    for column in [
        "statistical_alert",
        "spatial_alert",
        "lstm_ae_alert_original",
    ]:
        count = int(result[column].sum())
        print(f"  {column:26s}: " f"{count:,} " f"({count / len(result):.3%})")

    return result


def add_distance_to_nearest_anomaly(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()
    out["rows_from_nearest_anomaly"] = np.inf

    for station_id, group in out.groupby(
        "station_id",
        sort=False,
    ):

        group = group.sort_values("timestamp")

        anomaly_mask = group["is_anomaly"].astype(bool).to_numpy()

        distances = np.full(
            len(group),
            np.inf,
        )

        last_anomaly = None

        for i in range(len(group)):
            if anomaly_mask[i]:
                last_anomaly = i

            if last_anomaly is not None:
                distances[i] = i - last_anomaly

        next_anomaly = None

        for i in range(
            len(group) - 1,
            -1,
            -1,
        ):
            if anomaly_mask[i]:
                next_anomaly = i

            if next_anomaly is not None:
                distances[i] = min(
                    distances[i],
                    next_anomaly - i,
                )

        out.loc[
            group.index,
            "rows_from_nearest_anomaly",
        ] = distances

    return out


def build_fusion_variants(
    result: pd.DataFrame,
    lstm_calibration,
    multipliers: list[float],
) -> pd.DataFrame:

    result = result.copy()

    result["stat_spatial_and"] = result["statistical_alert"] & result["spatial_alert"]

    result["stat_spatial_or"] = result["statistical_alert"] | result["spatial_alert"]

    p99 = float(lstm_calibration.threshold_p99)

    for multiplier in multipliers:

        threshold = p99 * multiplier

        label = f"{multiplier:g}".replace(".", "p")

        alert_column = f"lstm_ae_alert_{label}x"

        result[alert_column] = result["lstm_ae_score"] > threshold

        result[f"stat_lstm_or_{label}x"] = (
            result["statistical_alert"] | result[alert_column]
        )

        result[f"spatial_lstm_or_{label}x"] = (
            result["spatial_alert"] | result[alert_column]
        )

        result[f"all_or_{label}x"] = (
            result["statistical_alert"] | result["spatial_alert"] | result[alert_column]
        )

        result[f"any_two_of_three_{label}x"] = (
            result[
                [
                    "statistical_alert",
                    "spatial_alert",
                    alert_column,
                ]
            ].sum(axis=1)
            >= 2
        )

        result[f"all_three_and_{label}x"] = (
            result["statistical_alert"] & result["spatial_alert"] & result[alert_column]
        )

    return result


def binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float | int]:

    truth = y_true.fillna(False).astype(bool)

    pred = y_pred.fillna(False).astype(bool)

    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "alert_count": int(pred.sum()),
        "alert_rate": float(pred.mean()),
    }


def event_metrics(
    df: pd.DataFrame,
    alert_column: str,
) -> dict[str, float | int]:

    anomalies = df[df["is_anomaly"]].copy()

    if anomalies.empty:
        return {
            "total_events": 0,
            "detected_events": 0,
            "event_recall": 0.0,
        }

    detected = anomalies.groupby("anomaly_id")[alert_column].any()

    total_events = len(detected)
    detected_events = int(detected.sum())

    return {
        "total_events": total_events,
        "detected_events": detected_events,
        "event_recall": (detected_events / total_events if total_events else 0.0),
    }


def buffer_clear_fpr(
    df: pd.DataFrame,
    alert_column: str,
) -> float:

    clean = df[(~df["is_anomaly"]) & (df["rows_from_nearest_anomaly"] > OVERLAP_BUFFER)]

    if clean.empty:
        return 0.0

    return float(clean[alert_column].fillna(False).astype(bool).mean())


def evaluate_methods(
    df: pd.DataFrame,
    methods: dict[str, str],
) -> pd.DataFrame:

    rows = []

    for method_name, column in methods.items():

        metrics = binary_metrics(
            df["is_anomaly"],
            df[column],
        )

        event = event_metrics(
            df,
            column,
        )

        clean_fpr = buffer_clear_fpr(
            df,
            column,
        )

        rows.append(
            {
                "method": method_name,
                **metrics,
                "event_recall": event["event_recall"],
                "detected_events": event["detected_events"],
                "total_events": event["total_events"],
                "buffer_clear_fpr": clean_fpr,
            }
        )

    summary = pd.DataFrame(rows)

    return summary


def evaluate_by_anomaly_type(
    df: pd.DataFrame,
    methods: dict[str, str],
) -> pd.DataFrame:

    anomalies = df[df["is_anomaly"]].copy()

    rows = []

    for anomaly_type, group in anomalies.groupby(
        "anomaly_type",
        dropna=False,
    ):

        row = {
            "anomaly_type": anomaly_type,
            "total_events": group["anomaly_id"].nunique(),
        }

        for method_name, column in methods.items():

            detected_events = group.groupby("anomaly_id")[column].any()

            total = len(detected_events)
            detected = int(detected_events.sum())

            row[f"{method_name}_recall"] = detected / total if total else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values("anomaly_type").reset_index(drop=True)


def evaluate_complementarity(
    df: pd.DataFrame,
    lstm_alert_column: str,
) -> dict[str, int | float]:

    stat = df["statistical_alert"].astype(bool)
    spatial = df["spatial_alert"].astype(bool)
    lstm = df[lstm_alert_column].astype(bool)

    baseline = stat | spatial
    combined = baseline | lstm

    anomalies = df["is_anomaly"].astype(bool)

    lstm_unique_tp = int((lstm & ~stat & ~spatial & anomalies).sum())

    newly_recovered_tp = int((combined & ~baseline & anomalies).sum())

    lstm_only_false_positive = int((lstm & ~stat & ~spatial & ~anomalies).sum())

    anomaly_events = (
        df.loc[
            anomalies,
            "anomaly_id",
        ]
        .dropna()
        .unique()
    )

    baseline_event_detected = {}
    combined_event_detected = {}
    lstm_only_event_detected = {}

    for event_id in anomaly_events:

        event = df[df["anomaly_id"] == event_id]

        baseline_event_detected[event_id] = bool(
            event["statistical_alert"].any() or event["spatial_alert"].any()
        )

        combined_event_detected[event_id] = bool(
            event[
                [
                    "statistical_alert",
                    "spatial_alert",
                    lstm_alert_column,
                ]
            ]
            .any(axis=1)
            .any()
        )

        lstm_only_event_detected[event_id] = bool(
            event[lstm_alert_column].any()
            and not event["statistical_alert"].any()
            and not event["spatial_alert"].any()
        )

    baseline_event_count = sum(baseline_event_detected.values())

    combined_event_count = sum(combined_event_detected.values())

    lstm_only_event_count = sum(lstm_only_event_detected.values())

    return {
        "baseline_event_detected": baseline_event_count,
        "combined_event_detected": combined_event_count,
        "new_events_recovered": (combined_event_count - baseline_event_count),
        "lstm_only_events": lstm_only_event_count,
        "lstm_only_tp_observations": lstm_unique_tp,
        "new_tp_observations": newly_recovered_tp,
        "lstm_only_false_positives": lstm_only_false_positive,
    }


def detector_overlap(
    df: pd.DataFrame,
    lstm_alert_column: str,
) -> pd.DataFrame:

    stat = df["statistical_alert"].astype(bool)
    spatial = df["spatial_alert"].astype(bool)
    lstm = df[lstm_alert_column].astype(bool)
    truth = df["is_anomaly"].astype(bool)

    masks = {
        "none": ~stat & ~spatial & ~lstm,
        "statistical_only": stat & ~spatial & ~lstm,
        "spatial_only": ~stat & spatial & ~lstm,
        "lstm_only": ~stat & ~spatial & lstm,
        "statistical_spatial": stat & spatial & ~lstm,
        "statistical_lstm": stat & lstm & ~spatial,
        "spatial_lstm": spatial & lstm & ~stat,
        "all_three": stat & spatial & lstm,
    }

    rows = []

    for name, mask in masks.items():

        observations = int(mask.sum())
        anomalies = int((mask & truth).sum())

        rows.append(
            {
                "category": name,
                "observations": observations,
                "true_anomalies": anomalies,
                "precision": (anomalies / observations if observations else 0.0),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(
    result: pd.DataFrame,
    all_metrics: pd.DataFrame,
    type_metrics: dict[str, pd.DataFrame],
    overlaps: dict[str, pd.DataFrame],
    complementarity: dict[str, pd.DataFrame],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics.to_csv(
        OUTPUT_DIR / "combined_metrics.csv",
        index=False,
    )

    for label, dataframe in type_metrics.items():
        dataframe.to_csv(
            OUTPUT_DIR / f"performance_by_anomaly_type_{label}.csv",
            index=False,
        )

    for label, dataframe in overlaps.items():
        dataframe.to_csv(
            OUTPUT_DIR / f"detector_overlap_{label}.csv",
            index=False,
        )

    for label, dataframe in complementarity.items():
        dataframe.to_csv(
            OUTPUT_DIR / f"lstm_complementarity_{label}.csv",
            index=False,
        )

    result.to_parquet(
        OUTPUT_DIR / "combined_full_results.parquet",
        index=False,
    )

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR / 'combined_metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'combined_full_results.parquet'}")


def print_metrics_table(
    metrics: pd.DataFrame,
) -> None:

    display = metrics[
        [
            "method",
            "alert_count",
            "alert_rate",
            "precision",
            "recall",
            "f1",
            (
                "false_positive_rate"
                if "false_positive_rate" in metrics.columns
                else "fpr"
            ),
            "event_recall",
            "buffer_clear_fpr",
        ]
    ].copy()

    if "fpr" in display.columns:
        display = display.rename(columns={"fpr": "false_positive_rate"})

    print("\n" + "=" * 110)
    print("COMBINED OBSERVATION + EVENT METRICS")
    print("=" * 110)

    print(
        display.to_string(
            index=False,
            formatters={
                "alert_rate": "{:.2%}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "false_positive_rate": "{:.4%}".format,
                "event_recall": "{:.2%}".format,
                "buffer_clear_fpr": "{:.2%}".format,
            },
        )
    )


def print_complementarity(
    label: str,
    complement: dict[str, int | float],
) -> None:

    print("\n" + "-" * 80)
    print(f"LSTM COMPLEMENTARITY — {label}")
    print("-" * 80)

    print(
        f"  Baseline stat+spatial detected events: "
        f"{complement['baseline_event_detected']}"
    )

    print(
        f"  Combined detected events:              "
        f"{complement['combined_event_detected']}"
    )

    print(
        f"  New events recovered by LSTM:          "
        f"{complement['new_events_recovered']}"
    )

    print(
        f"  LSTM-only anomaly events:               "
        f"{complement['lstm_only_events']}"
    )

    print(
        f"  LSTM-only TP observations:              "
        f"{complement['lstm_only_tp_observations']}"
    )

    print(
        f"  New TP observations from LSTM:         "
        f"{complement['new_tp_observations']}"
    )

    print(
        f"  LSTM-only false-positive observations: "
        f"{complement['lstm_only_false_positives']}"
    )


def generate_comparison_plot(
    metrics: pd.DataFrame,
) -> Path:

    plot_df = metrics.copy()

    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(
        x - width,
        plot_df["precision"],
        width,
        label="Precision",
    )

    ax.bar(
        x,
        plot_df["recall"],
        width,
        label="Recall",
    )

    ax.bar(
        x + width,
        plot_df["f1"],
        width,
        label="F1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["method"],
        rotation=35,
        ha="right",
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("SkyGuard Three-Detector Baseline Comparison")

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "three_detector_comparison.png"

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def main() -> None:

    args = parse_args()

    multipliers = [
        float(value.strip())
        for value in args.lstm_multipliers.split(",")
        if value.strip()
    ]

    if not multipliers:
        raise ValueError("At least one LSTM multiplier is required.")

    print("\n" + "=" * 80)
    print("SKYGUARD — THREE-DETECTOR COMBINED EVALUATOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    historical_df, clean_present_df = load_datasets()

    (
        stat_calibration,
        stat_config,
        spatial_calibration,
        spatial_config,
    ) = calibrate_stat_spatial(historical_df)

    (
        lstm_calibration,
        lstm_config,
    ) = load_lstm_calibration()

    corrupted_df, injection_log = inject_shared_anomalies(clean_present_df)

    result = run_detectors(
        corrupted_df,
        stat_calibration,
        stat_config,
        spatial_calibration,
        spatial_config,
        lstm_calibration,
        lstm_config,
    )

    result = add_distance_to_nearest_anomaly(result)

    result = build_fusion_variants(
        result,
        lstm_calibration,
        multipliers,
    )

    methods = {
        "statistical": "statistical_alert",
        "spatial": "spatial_alert",
        "stat_spatial_AND": "stat_spatial_and",
        "stat_spatial_OR": "stat_spatial_or",
    }

    for multiplier in multipliers:

        label = f"{multiplier:g}".replace(".", "p")

        lstm_column = f"lstm_ae_alert_{label}x"

        methods[f"lstm_{label}x"] = lstm_column

        methods[f"stat_lstm_OR_{label}x"] = f"stat_lstm_or_{label}x"

        methods[f"spatial_lstm_OR_{label}x"] = f"spatial_lstm_or_{label}x"

        methods[f"all_three_OR_{label}x"] = f"all_or_{label}x"

        methods[f"any_two_of_three_{label}x"] = f"any_two_of_three_{label}x"

        methods[f"all_three_AND_{label}x"] = f"all_three_and_{label}x"

    metrics = evaluate_methods(
        result,
        methods,
    )

    print_metrics_table(metrics)

    type_metrics = {}

    for multiplier in multipliers:

        label = f"{multiplier:g}".replace(".", "p")

        selected_methods = {
            "statistical": "statistical_alert",
            "spatial": "spatial_alert",
            f"lstm_{label}x": f"lstm_ae_alert_{label}x",
            f"stat_spatial_OR": "stat_spatial_or",
            f"all_three_OR_{label}x": f"all_or_{label}x",
            f"any_two_of_three_{label}x": f"any_two_of_three_{label}x",
        }

        type_metrics[label] = evaluate_by_anomaly_type(
            result,
            selected_methods,
        )

    complementarity = {}
    overlaps = {}

    for multiplier in multipliers:

        label = f"{multiplier:g}".replace(".", "p")

        lstm_column = f"lstm_ae_alert_{label}x"

        complement = evaluate_complementarity(
            result,
            lstm_column,
        )

        print_complementarity(
            f"{label}x P99",
            complement,
        )

        complementarity[label] = pd.DataFrame([complement])

        overlaps[label] = detector_overlap(
            result,
            lstm_column,
        )

    for label, dataframe in type_metrics.items():

        print("\n" + "=" * 110)
        print(f"EVENT RECALL BY ANOMALY TYPE — " f"LSTM {label}x P99")
        print("=" * 110)

        print(
            dataframe.to_string(
                index=False,
                formatters={
                    column: "{:.2%}".format
                    for column in dataframe.columns
                    if column.endswith("_recall")
                },
            )
        )

    save_outputs(
        result=result,
        all_metrics=metrics,
        type_metrics=type_metrics,
        overlaps=overlaps,
        complementarity=complementarity,
    )

    plot_path = generate_comparison_plot(metrics)

    print(f"\n✓ Comparison plot saved: {plot_path}")

    print("\n" + "=" * 80)
    print("COMBINED EVALUATION COMPLETE")
    print("=" * 80)

    print("\nExperiment design:")

    print("  Same injected benchmark: YES")

    print("  Same statistical output: YES")

    print("  Same spatial output:     YES")

    print("  One LSTM inference pass: YES")

    print("  LSTM retraining:         NO")

    print("  Persistence:             NO")

    print("  Fusion weight tuning:    NO")

    print("\nLSTM multipliers tested:")

    for multiplier in multipliers:
        threshold = lstm_calibration.threshold_p99 * multiplier

        print(f"  {multiplier:.2f}x P99 = " f"{threshold:.6f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
