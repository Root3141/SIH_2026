"""Evaluate the spatial detector."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyguard.config.paths import (
    PRESENT_PARQUET,
    RESULTS_DIR,
    SYNTHETIC_DATA_DIR,
    SPATIAL_EVAL_INJECTED_PARQUET,
)
from skyguard.detectors.spatial import (
    SpatialConfig,
    calibrate_spatial_detector,
    run_spatial_detector,
)
from skyguard.simulation.anomaly_injector import AnomalyConfig, inject_anomalies

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

HISTORICAL_PARQUET = DATA_DIR / "ncr_weather_historical.parquet"
PRESENT_PARQUET = DATA_DIR / "ncr_weather_2026_present.parquet"
OUTPUT_DIR = RESULTS_DIR / "spatial"

VARIABLES = ["temperature", "humidity", "pressure"]
RANDOM_SEED = 42


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading datasets...\n")

    if not HISTORICAL_PARQUET.exists():
        raise FileNotFoundError(f"Historical dataset not found:\n{HISTORICAL_PARQUET}")
    if not PRESENT_PARQUET.exists():
        raise FileNotFoundError(f"Evaluation dataset not found:\n{PRESENT_PARQUET}")

    historical_df = pd.read_parquet(HISTORICAL_PARQUET)
    present_df = pd.read_parquet(PRESENT_PARQUET)

    for df in (historical_df, present_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("Historical calibration data:")
    print(f"  Rows:     {len(historical_df):,}")
    print(f"  Stations: {historical_df['station_id'].nunique()}")
    print(
        f"  Period:   {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}"
    )

    print("\nUnseen evaluation data:")
    print(f"  Rows:     {len(present_df):,}")
    print(f"  Stations: {present_df['station_id'].nunique()}")
    print(
        f"  Period:   {present_df['timestamp'].min()} to {present_df['timestamp'].max()}"
    )

    return historical_df, present_df


def calibrate_detector(historical_df: pd.DataFrame) -> tuple[object, SpatialConfig]:
    print("\n" + "=" * 70)
    print("CALIBRATING SPATIAL DETECTOR")
    print("=" * 70)

    config = SpatialConfig()

    station_coords = (
        historical_df[["station_id", "latitude", "longitude"]]
        .drop_duplicates()
        .set_index("station_id")
        .apply(lambda row: (float(row["latitude"]), float(row["longitude"])), axis=1)
        .to_dict()
    )

    calibration = calibrate_spatial_detector(
        historical_df,
        VARIABLES,
        station_coords,
        config,
    )

    print("\n✓ Spatial calibration complete")
    return calibration, config


def inject_evaluation_anomalies(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("INJECTING SYNTHETIC ANOMALIES")
    print("=" * 70)

    config = AnomalyConfig(anomaly_rate=0.02, random_seed=RANDOM_SEED)
    corrupted_df, injection_log = inject_anomalies(clean_df, VARIABLES, config)

    if "is_anomaly" not in corrupted_df.columns:
        corrupted_df = corrupted_df.copy()
        corrupted_df["is_anomaly"] = (
            corrupted_df.get("synthetic_anomaly", False).fillna(False).astype(bool)
        )
        corrupted_df["anomaly_id"] = corrupted_df.get("synthetic_anomaly_event_id")
        corrupted_df["anomaly_type"] = corrupted_df.get("synthetic_anomaly_type")
        corrupted_df["anomaly_variable"] = corrupted_df.get("synthetic_anomaly_variable")
        corrupted_df["anomaly_severity"] = corrupted_df.get("synthetic_anomaly_severity")

    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corrupted_df.to_parquet(SPATIAL_EVAL_INJECTED_PARQUET, index=False)

    print(f"\n✓ Injected dataset saved to: {SPATIAL_EVAL_INJECTED_PARQUET}")

    print("\n✓ Anomaly injection complete")

    if "is_anomaly" in corrupted_df.columns:
        anomaly_count = corrupted_df["is_anomaly"].sum()
        anomaly_rate = anomaly_count / len(corrupted_df) * 100
        print(f"  Total observations:      {len(corrupted_df):,}")
        print(f"  Anomalous observations:  {anomaly_count:,}")
        print(f"  Actual anomaly rate:     {anomaly_rate:.3f}%")

    if injection_log is not None:
        print(f"  Injected anomaly events: {len(injection_log):,}")

    return corrupted_df, injection_log


def run_detector(
    evaluation_df: pd.DataFrame, calibration, config: SpatialConfig
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("RUNNING SPATIAL DETECTOR")
    print("=" * 70)

    result_df = run_spatial_detector(evaluation_df, VARIABLES, calibration, config)

    alert_count = result_df["spatial_alert"].sum()
    alert_rate = alert_count / len(result_df) * 100

    print("\n✓ Spatial detection complete")
    print(f"  Total observations: {len(result_df):,}")
    print(f"  Spatial alerts:     {alert_count:,}")
    print(f"  Alert rate:         {alert_rate:.3f}%")

    return result_df


def calculate_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.fillna(False).astype(bool)
    y_pred = y_pred.fillna(False).astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
    }


def evaluate_observation_level(result_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("OBSERVATION-LEVEL PERFORMANCE")
    print("=" * 70)

    if "is_anomaly" not in result_df.columns:
        raise ValueError(
            "Ground-truth column 'is_anomaly' not found in evaluation data."
        )

    metrics = calculate_binary_metrics(
        result_df["is_anomaly"], result_df["spatial_alert"]
    )

    metrics_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )

    print(f"\nTrue Positives:  {metrics['true_positive']:,}")
    print(f"False Positives: {metrics['false_positive']:,}")
    print(f"False Negatives: {metrics['false_negative']:,}")
    print(f"True Negatives:  {metrics['true_negative']:,}")

    print("\nClassification metrics:")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1 Score:            {metrics['f1_score']:.4f}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")

    return metrics_df


def evaluate_event_level(result_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("EVENT-LEVEL PERFORMANCE")
    print("=" * 70)

    if "anomaly_id" not in result_df.columns:
        print("\n⚠ anomaly_id column not found. Skipping event-level evaluation.")
        return pd.DataFrame()

    anomaly_rows = result_df[result_df["is_anomaly"].fillna(False)].copy()
    if anomaly_rows.empty:
        print("\nNo injected anomaly observations found.")
        return pd.DataFrame()

    event_stats = (
        anomaly_rows.groupby("anomaly_id")
        .agg(
            anomaly_type=("anomaly_type", "first"),
            variable=("anomaly_variable", "first"),
            duration=("anomaly_id", "size"),
            detected=("spatial_alert", "any"),
            observations_detected=("spatial_alert", "sum"),
        )
        .reset_index()
    )

    total_events = len(event_stats)
    detected_events = int(event_stats["detected"].sum())
    event_recall = detected_events / total_events if total_events > 0 else 0.0

    print(f"\nTotal anomaly events:    {total_events:,}")
    print(f"Detected anomaly events: {detected_events:,}")
    print(f"Missed anomaly events:   {total_events - detected_events:,}")
    print(f"Event-level recall:      {event_recall:.4f}")

    return event_stats


def evaluate_by_anomaly_type(result_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PERFORMANCE BY ANOMALY TYPE")
    print("=" * 70)

    if "anomaly_type" not in result_df.columns:
        print("\n⚠ anomaly_type column not found.")
        return pd.DataFrame()

    anomaly_rows = result_df[result_df["is_anomaly"].fillna(False)].copy()
    rows = []

    for anomaly_type, group in anomaly_rows.groupby("anomaly_type"):
        total = len(group)
        detected = int(group["spatial_alert"].sum())
        recall = detected / total if total > 0 else 0.0
        rows.append(
            {
                "anomaly_type": anomaly_type,
                "total_observations": total,
                "detected": detected,
                "missed": total - detected,
                "recall": recall,
                "recall_percent": recall * 100,
            }
        )

    summary = (
        pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)
    )

    print(
        summary.to_string(
            index=False,
            formatters={"recall": "{:.4f}".format, "recall_percent": "{:.2f}".format},
        )
    )
    return summary


def evaluate_by_variable(result_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PERFORMANCE BY VARIABLE")
    print("=" * 70)

    missing = [
        variable
        for variable in VARIABLES
        if f"{variable}_spatial_anomaly" not in result_df.columns
    ]
    if missing:
        raise ValueError(f"Missing detector columns: {missing}")

    rows = []
    for variable in VARIABLES:
        y_true = result_df["is_anomaly"].fillna(False) & (
            result_df["anomaly_variable"] == variable
        )
        y_pred = result_df[f"{variable}_spatial_anomaly"].fillna(False)
        metrics = calculate_binary_metrics(y_true, y_pred)
        rows.append({"variable": variable, **metrics})
    summary = pd.DataFrame(rows)
    print(
        summary[
            [
                "variable",
                "true_positive",
                "false_positive",
                "false_negative",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
            ]
        ].to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
                "false_positive_rate": "{:.4f}".format,
            },
        )
    )
    return summary


def analyze_false_positives(result_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 70)

    clean_rows = result_df[~result_df["is_anomaly"].fillna(False)].copy()
    false_positives = clean_rows[clean_rows["spatial_alert"].fillna(False)].copy()

    total_clean = len(clean_rows)
    fp_count = len(false_positives)
    fp_rate = fp_count / total_clean * 100 if total_clean > 0 else 0.0

    print(f"\nClean observations: {total_clean:,}")
    print(f"False positives:    {fp_count:,}")
    print(f"False positive rate: {fp_rate:.4f}%")

    if false_positives.empty:
        return pd.DataFrame()

    station_summary = (
        false_positives.groupby("station_id")
        .size()
        .rename("false_positive_count")
        .reset_index()
        .sort_values("false_positive_count", ascending=False)
    )

    print("\nFalse positives by station:")
    print(station_summary.to_string(index=False))

    return station_summary


def analyze_severity_distribution(result_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("SPATIAL SEVERITY DISTRIBUTION")
    print("=" * 70)

    severity_column = "spatial_severity_level"
    if severity_column not in result_df.columns:
        print("\n⚠ Severity column not found.")
        return pd.DataFrame()

    summary = (
        result_df.groupby(severity_column)
        .agg(
            observations=("station_id", "size"),
            anomalies=("is_anomaly", "sum"),
            alerts=("spatial_alert", "sum"),
        )
        .reset_index()
    )

    summary["anomaly_rate_percent"] = (
        summary["anomalies"] / summary["observations"] * 100
    )
    print(
        summary.to_string(
            index=False, formatters={"anomaly_rate_percent": "{:.2f}".format}
        )
    )
    return summary


def run_regional_event_test(
    clean_df: pd.DataFrame, calibration, config: SpatialConfig
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("REGIONAL EVENT SANITY TEST")
    print("=" * 70)

    test_df = clean_df.copy()
    timestamps = sorted(test_df["timestamp"].unique())

    if len(timestamps) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_SEED)
    selected_timestamp = timestamps[
        rng.integers(low=len(timestamps) // 4, high=3 * len(timestamps) // 4)
    ]
    variable = "temperature"
    regional_offset = 6.0

    mask = test_df["timestamp"] == selected_timestamp
    station_count = mask.sum()

    test_df.loc[mask, variable] += regional_offset

    print(
        f"\nInjected regional event:\n  Timestamp: {selected_timestamp}\n  Variable: {variable}\n  Shared offset: +{regional_offset}\n  Stations affected: {station_count}"
    )

    result_df = run_spatial_detector(test_df, VARIABLES, calibration, config)

    event_rows = result_df[result_df["timestamp"] == selected_timestamp].copy()
    alert_count = event_rows["spatial_alert"].sum()

    print(f"\nSpatial alerts during regional event: {alert_count}/{len(event_rows)}")

    if alert_count == 0:
        print("✓ PASS — regional event was not misclassified.")
    else:
        print(
            "⚠ WARNING — some stations were flagged during the shared regional event."
        )

    return event_rows


def generate_plots(
    result_df: pd.DataFrame, anomaly_type_summary: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Spatial Detector Evaluation", fontsize=16, fontweight="bold")

    y_true = result_df["is_anomaly"].fillna(False).astype(bool)
    y_pred = result_df["spatial_alert"].fillna(False).astype(bool)

    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    tn = (~y_true & ~y_pred).sum()

    matrix = np.array([[tn, fp], [fn, tp]])

    image = axes[0, 0].imshow(matrix)
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Normal", "Anomaly"])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_yticklabels(["Normal", "Anomaly"])
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Ground Truth")
    axes[0, 0].set_title("Confusion Matrix")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(
                j,
                i,
                f"{labels[i][j]}\n{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
            )

    if not anomaly_type_summary.empty:
        axes[0, 1].bar(
            anomaly_type_summary["anomaly_type"], anomaly_type_summary["recall_percent"]
        )
        axes[0, 1].set_title("Recall by Anomaly Type")
        axes[0, 1].set_xlabel("Anomaly Type")
        axes[0, 1].set_ylabel("Recall (%)")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].grid(axis="y", alpha=0.3)

    station_alerts = (
        result_df.groupby("station_id")["spatial_alert"].mean().sort_values() * 100
    )
    axes[1, 0].barh(station_alerts.index, station_alerts.values)
    axes[1, 0].set_title("Spatial Alert Rate by Station")
    axes[1, 0].set_xlabel("Alert Rate (%)")
    axes[1, 0].grid(axis="x", alpha=0.3)

    if "spatial_severity_level" in result_df.columns:
        severity_counts = result_df["spatial_severity_level"].value_counts()
        axes[1, 1].bar(severity_counts.index.astype(str), severity_counts.values)
        axes[1, 1].set_title("Spatial Severity Distribution")
        axes[1, 1].set_xlabel("Severity")
        axes[1, 1].set_ylabel("Observations")
        axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "spatial_detector_evaluation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Evaluation plot saved:\n  {output_path}")


def save_results(
    result_df: pd.DataFrame,
    observation_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    anomaly_type_summary: pd.DataFrame,
    variable_summary: pd.DataFrame,
    false_positive_summary: pd.DataFrame,
    severity_summary: pd.DataFrame,
    regional_event_results: pd.DataFrame,
    output_dir: Path,
    injection_log: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if injection_log is not None and not injection_log.empty:
        injection_log.to_csv(output_dir / "injection_log.csv", index=False)

    files = {
        "observation_metrics.csv": observation_metrics,
        "event_metrics.csv": event_metrics,
        "performance_by_anomaly_type.csv": anomaly_type_summary,
        "performance_by_variable.csv": variable_summary,
        "false_positive_analysis.csv": false_positive_summary,
        "severity_distribution.csv": severity_summary,
        "regional_event_test.csv": regional_event_results,
    }

    for filename, dataframe in files.items():
        if dataframe is not None and not dataframe.empty:
            path = output_dir / filename
            dataframe.to_csv(path, index=False)
            print(f"✓ Saved: {path}")

    anomaly_results = result_df[
        result_df["is_anomaly"].fillna(False) | result_df["spatial_alert"].fillna(False)
    ].copy()
    anomaly_results_path = output_dir / "anomaly_predictions.csv"
    anomaly_results.to_csv(anomaly_results_path, index=False)
    print(f"✓ Saved: {anomaly_results_path}")

    full_results_path = output_dir / "full_results.parquet"
    result_df.to_parquet(full_results_path, index=False)
    print(f"✓ Saved: {full_results_path}")


def main() -> None:
    print("\n" + "=" * 70)
    print("SPATIAL DETECTOR EVALUATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    historical_df, clean_present_df = load_datasets()
    calibration, spatial_config = calibrate_detector(historical_df)
    corrupted_df, injection_log = inject_evaluation_anomalies(clean_present_df)

    result_df = run_detector(corrupted_df, calibration, spatial_config)

    observation_metrics = evaluate_observation_level(result_df)
    event_metrics = evaluate_event_level(result_df)
    anomaly_type_summary = evaluate_by_anomaly_type(result_df)
    variable_summary = evaluate_by_variable(result_df)
    false_positive_summary = analyze_false_positives(result_df)
    severity_summary = analyze_severity_distribution(result_df)
    regional_event_results = run_regional_event_test(
        clean_present_df, calibration, spatial_config
    )

    generate_plots(result_df, anomaly_type_summary, OUTPUT_DIR)

    save_results(
        result_df=result_df,
        observation_metrics=observation_metrics,
        event_metrics=event_metrics,
        anomaly_type_summary=anomaly_type_summary,
        variable_summary=variable_summary,
        false_positive_summary=false_positive_summary,
        severity_summary=severity_summary,
        regional_event_results=regional_event_results,
        output_dir=OUTPUT_DIR,
        injection_log=injection_log,
    )

    print("\n" + "=" * 70)
    print(
        f"Spatial evaluation complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("Results saved to:")
    print(OUTPUT_DIR)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
