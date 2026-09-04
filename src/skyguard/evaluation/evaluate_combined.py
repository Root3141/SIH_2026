"""
Simple combined evaluator for the statistical and spatial detectors.

Intended only for debugging and rough comparison, not rigorous fusion evaluation.
Both detectors evaluate the same synthetically corrupted data.

Reports individual performance, detector overlap, simple AND/OR fusion baselines,
event-level metrics, anomaly-type performance, and regional-event sanity checks.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

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
from skyguard.simulation.anomaly_injector import AnomalyConfig, inject_anomalies


VARIABLES = ["temperature", "humidity", "pressure"]
RANDOM_SEED = 42
OUTPUT_DIR = RESULTS_DIR / "statistical_spatial"
INJECTED_PARQUET = SYNTHETIC_DATA_DIR / "statistical_spatial_evaluation_injected.parquet"

KEY_COLUMNS = ["station_id", "timestamp"]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78)
    print("LOADING DATASETS")
    print("=" * 78)

    if not HISTORICAL_PARQUET.exists():
        raise FileNotFoundError(f"Historical dataset not found:\n{HISTORICAL_PARQUET}")
    if not PRESENT_PARQUET.exists():
        raise FileNotFoundError(f"Evaluation dataset not found:\n{PRESENT_PARQUET}")

    historical_df = pd.read_parquet(HISTORICAL_PARQUET).copy()
    present_df = pd.read_parquet(PRESENT_PARQUET).copy()

    for df in (historical_df, present_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    historical_df = historical_df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    present_df = present_df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    print("\nHistorical calibration data:")
    print(f"  Rows:     {len(historical_df):,}")
    print(f"  Stations: {historical_df['station_id'].nunique():,}")
    print(f"  Period:   {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")

    print("\nUnseen evaluation data:")
    print(f"  Rows:     {len(present_df):,}")
    print(f"  Stations: {present_df['station_id'].nunique():,}")
    print(f"  Period:   {present_df['timestamp'].min()} to {present_df['timestamp'].max()}")

    return historical_df, present_df


def inject_shared_anomalies(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78)
    print("INJECTING ONE SHARED SYNTHETIC BENCHMARK")
    print("=" * 78)

    config = AnomalyConfig(anomaly_rate=0.02, random_seed=RANDOM_SEED)
    corrupted_df, injection_log = inject_anomalies(clean_df.copy(), VARIABLES, config)
    corrupted_df = corrupted_df.copy()

    if "is_anomaly" not in corrupted_df.columns:
        corrupted_df["is_anomaly"] = (
            corrupted_df.get("synthetic_anomaly", pd.Series(False, index=corrupted_df.index))
            .fillna(False)
            .astype(bool)
        )
    if "anomaly_id" not in corrupted_df.columns:
        corrupted_df["anomaly_id"] = corrupted_df.get("synthetic_anomaly_event_id")
    if "anomaly_type" not in corrupted_df.columns:
        corrupted_df["anomaly_type"] = corrupted_df.get("synthetic_anomaly_type")
    if "anomaly_variable" not in corrupted_df.columns:
        corrupted_df["anomaly_variable"] = corrupted_df.get("synthetic_anomaly_variable")
    if "anomaly_severity" not in corrupted_df.columns:
        corrupted_df["anomaly_severity"] = corrupted_df.get("synthetic_anomaly_severity")

    required = {"is_anomaly", "anomaly_id", "anomaly_type", "anomaly_variable", "anomaly_severity"}
    missing = sorted(required - set(corrupted_df.columns))
    if missing:
        raise ValueError(f"Missing ground-truth columns after anomaly injection: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corrupted_df.to_parquet(INJECTED_PARQUET, index=False)

    anomaly_count = int(corrupted_df["is_anomaly"].fillna(False).sum())
    rate = anomaly_count / len(corrupted_df) * 100 if len(corrupted_df) else 0.0
    event_count = int(corrupted_df.loc[corrupted_df["is_anomaly"], "anomaly_id"].nunique())

    print(f"  Total observations: {len(corrupted_df):,}")
    print(f"  Anomalous observations: {anomaly_count:,}")
    print(f"  Actual anomaly rate: {rate:.3f}%")
    print(f"  Anomaly events: {event_count:,}")
    print(f"  Seed: {RANDOM_SEED}")
    print(f"  Saved shared benchmark: {INJECTED_PARQUET}")

    return corrupted_df.sort_values(KEY_COLUMNS).reset_index(drop=True), injection_log


def calibrate_detectors(historical_df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("CALIBRATING BOTH DETECTORS ON HISTORICAL DATA ONLY")
    print("=" * 78)

    stat_config = StatisticalConfig()
    stat_calibration = calibrate_statistical_detector(historical_df, VARIABLES, stat_config)

    station_coords = (
        historical_df[["station_id", "latitude", "longitude"]]
        .drop_duplicates()
        .set_index("station_id")
        .apply(lambda row: (float(row["latitude"]), float(row["longitude"])), axis=1)
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
    print(f"  ROC percentile: {stat_config.roc_percentile}")
    print(f"  Z-score threshold: {stat_config.zscore_threshold}")
    print(f"  Z-score windows: {stat_config.zscore_windows}")
    print(f"  Persistence window: {stat_config.persistence_window}")

    print("\nSpatial configuration:")
    print(f"  k neighbors: {spatial_config.k_neighbors}")
    print(f"  IDW power: {spatial_config.idw_power}")
    print(f"  Suspicious percentile: {spatial_config.suspicious_percentile}")
    print(f"  Anomaly percentile: {spatial_config.anomaly_percentile}")

    return stat_calibration, stat_config, spatial_calibration, spatial_config


def run_both_detectors(
    evaluation_df: pd.DataFrame,
    stat_calibration,
    stat_config: StatisticalConfig,
    spatial_calibration,
    spatial_config: SpatialConfig,
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("RUNNING STATISTICAL + SPATIAL DETECTORS")
    print("=" * 78)

    stat_result = run_statistical_detector(
        evaluation_df.copy(), VARIABLES, stat_calibration, stat_config
    ).sort_values(KEY_COLUMNS).reset_index(drop=True)
    spatial_result = run_spatial_detector(
        evaluation_df.copy(), VARIABLES, spatial_calibration, spatial_config
    ).sort_values(KEY_COLUMNS).reset_index(drop=True)

    if len(stat_result) != len(spatial_result):
        raise ValueError("Statistical and spatial outputs have different row counts.")
    if not stat_result[KEY_COLUMNS].reset_index(drop=True).equals(
        spatial_result[KEY_COLUMNS].reset_index(drop=True)
    ):
        raise ValueError("Statistical and spatial outputs do not share identical station/timestamp keys.")

    # Start from the statistical result. Add only spatial-specific columns to avoid
    # duplicating measurements and ground-truth fields.
    result = stat_result.copy()
    spatial_columns = [
        column
        for column in spatial_result.columns
        if column in {"spatial_alert", "spatial_severity_level"}
        or column.endswith("_spatial_anomaly")
        or column.endswith("_spatial_suspicious")
        or "neighbor" in column.lower()
        or column.startswith("spatial_")
    ]
    spatial_columns = [column for column in dict.fromkeys(spatial_columns) if column not in KEY_COLUMNS]

    result = result.join(
        spatial_result.set_index(KEY_COLUMNS)[spatial_columns],
        on=KEY_COLUMNS,
    )

    if "spatial_alert" not in result.columns:
        raise ValueError("Spatial detector output column 'spatial_alert' not found.")
    if "statistical_alert" not in result.columns:
        raise ValueError("Statistical detector output column 'statistical_alert' not found.")

    result["statistical_alert"] = result["statistical_alert"].fillna(False).astype(bool)
    result["spatial_alert"] = result["spatial_alert"].fillna(False).astype(bool)
    result["both_alert"] = result["statistical_alert"] & result["spatial_alert"]
    result["either_alert"] = result["statistical_alert"] | result["spatial_alert"]
    result["statistical_only_alert"] = result["statistical_alert"] & ~result["spatial_alert"]
    result["spatial_only_alert"] = result["spatial_alert"] & ~result["statistical_alert"]
    result["neither_alert"] = ~result["statistical_alert"] & ~result["spatial_alert"]
    result["is_anomaly"] = result["is_anomaly"].fillna(False).astype(bool)

    print(f"\nRows scored: {len(result):,}")
    for column in ["statistical_alert", "spatial_alert", "both_alert", "either_alert"]:
        count = int(result[column].sum())
        print(f"  {column}: {count:,} ({count / len(result) * 100:.4f}%)")

    return result


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    y_true = y_true.fillna(False).astype(bool)
    y_pred = y_pred.fillna(False).astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "alert_count": int(y_pred.sum()),
        "alert_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }


def evaluate_methods(result_df: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "statistical": "statistical_alert",
        "spatial": "spatial_alert",
        "both_and": "both_alert",
        "either_or": "either_alert",
        "statistical_only": "statistical_only_alert",
        "spatial_only": "spatial_only_alert",
    }

    rows = []
    for method, column in methods.items():
        metrics = binary_metrics(result_df["is_anomaly"], result_df[column])
        rows.append({"method": method, **metrics})

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("OBSERVATION-LEVEL COMPARISON")
    print("=" * 78)
    display = summary[
        [
            "method",
            "alert_count",
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        ]
    ].copy()
    print(
        display.to_string(
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


def evaluate_overlap(result_df: pd.DataFrame) -> pd.DataFrame:
    y_true = result_df["is_anomaly"]
    stat = result_df["statistical_alert"]
    spatial = result_df["spatial_alert"]

    masks = {
        "neither": ~stat & ~spatial,
        "statistical_only": stat & ~spatial,
        "spatial_only": ~stat & spatial,
        "both": stat & spatial,
    }

    rows = []
    for name, mask in masks.items():
        count = int(mask.sum())
        tp = int((mask & y_true).sum())
        fp = int((mask & ~y_true).sum())
        precision = tp / count if count else 0.0
        rows.append(
            {
                "overlap_category": name,
                "observations": count,
                "true_anomalies": tp,
                "false_positives": fp,
                "precision": precision,
                "rate_percent": count / len(result_df) * 100 if len(result_df) else 0.0,
            }
        )

    overlap = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("DETECTOR OVERLAP / COMPLEMENTARITY")
    print("=" * 78)
    print(
        overlap.to_string(
            index=False,
            formatters={"precision": "{:.4f}".format, "rate_percent": "{:.4f}".format},
        )
    )

    both_tp = int((masks["both"] & y_true).sum())
    stat_tp = int((stat & y_true).sum())
    spatial_tp = int((spatial & y_true).sum())
    union_tp = int(((stat | spatial) & y_true).sum())

    stat_added = int((masks["statistical_only"] & y_true).sum())
    spatial_added = int((masks["spatial_only"] & y_true).sum())

    print("\nComplementarity summary:")
    print(f"  True anomalies caught by both:       {both_tp:,}")
    print(f"  True anomalies caught by statistical only: {stat_added:,}")
    print(f"  True anomalies caught by spatial only:     {spatial_added:,}")
    print(f"  Union true positives:                 {union_tp:,}")
    print(f"  Union recall gain over statistical:   {(union_tp - stat_tp):,} observations")
    print(f"  Union recall gain over spatial:       {(union_tp - spatial_tp):,} observations")

    return overlap


def evaluate_event_level(result_df: pd.DataFrame) -> pd.DataFrame:
    anomaly_rows = result_df[result_df["is_anomaly"]].copy()
    if anomaly_rows.empty or "anomaly_id" not in anomaly_rows.columns:
        return pd.DataFrame()

    methods = ["statistical_alert", "spatial_alert", "both_alert", "either_alert"]
    grouped = anomaly_rows.groupby("anomaly_id")

    event = grouped.agg(
        anomaly_type=("anomaly_type", "first"),
        variable=("anomaly_variable", "first"),
        severity=("anomaly_severity", "first"),
        duration=("anomaly_id", "size"),
    ).reset_index()

    for method in methods:
        detected = grouped[method].any().rename(f"{method}_detected")
        count = grouped[method].sum().rename(f"{method}_observations_detected")
        event = event.merge(detected, on="anomaly_id").merge(count, on="anomaly_id")

    print("\n" + "=" * 78)
    print("EVENT-LEVEL PERFORMANCE")
    print("=" * 78)

    rows = []
    total_events = len(event)
    for method in methods:
        detected_count = int(event[f"{method}_detected"].sum())
        recall = detected_count / total_events if total_events else 0.0
        rows.append(
            {
                "method": method.replace("_alert", ""),
                "total_events": total_events,
                "detected_events": detected_count,
                "missed_events": total_events - detected_count,
                "event_recall": recall,
                "event_recall_percent": recall * 100,
            }
        )

    summary = pd.DataFrame(rows)
    print(
        summary.to_string(
            index=False,
            formatters={"event_recall": "{:.4f}".format, "event_recall_percent": "{:.2f}".format},
        )
    )
    return event


def evaluate_by_anomaly_type(result_df: pd.DataFrame) -> pd.DataFrame:
    anomaly_rows = result_df[result_df["is_anomaly"]].copy()
    if anomaly_rows.empty:
        return pd.DataFrame()

    methods = ["statistical_alert", "spatial_alert", "both_alert", "either_alert"]
    rows = []
    for anomaly_type, group in anomaly_rows.groupby("anomaly_type", dropna=False):
        row = {"anomaly_type": anomaly_type, "total_observations": len(group)}
        for method in methods:
            detected = int(group[method].sum())
            recall = detected / len(group) if len(group) else 0.0
            row[f"{method}_detected"] = detected
            row[f"{method}_recall"] = recall
            row[f"{method}_recall_percent"] = recall * 100
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("anomaly_type").reset_index(drop=True)

    print("\n" + "=" * 78)
    print("RECALL BY ANOMALY TYPE")
    print("=" * 78)
    display_columns = [
        "anomaly_type",
        "total_observations",
        "statistical_alert_recall_percent",
        "spatial_alert_recall_percent",
        "both_alert_recall_percent",
        "either_alert_recall_percent",
    ]
    print(
        summary[display_columns].to_string(
            index=False,
            formatters={column: "{:.2f}".format for column in display_columns[2:]},
        )
    )
    return summary


def evaluate_by_variable(result_df: pd.DataFrame) -> pd.DataFrame:
    anomaly_rows = result_df[result_df["is_anomaly"]].copy()
    if anomaly_rows.empty:
        return pd.DataFrame()

    methods = ["statistical_alert", "spatial_alert", "both_alert", "either_alert"]
    rows = []
    for variable in VARIABLES:
        group = anomaly_rows[anomaly_rows["anomaly_variable"] == variable]
        row = {"variable": variable, "anomaly_observations": len(group)}
        for method in methods:
            detected = int(group[method].sum())
            recall = detected / len(group) if len(group) else 0.0
            row[f"{method}_detected"] = detected
            row[f"{method}_recall"] = recall
            row[f"{method}_recall_percent"] = recall * 100
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("RECALL BY ANOMALY VARIABLE")
    print("=" * 78)
    display_columns = [
        "variable",
        "anomaly_observations",
        "statistical_alert_recall_percent",
        "spatial_alert_recall_percent",
        "both_alert_recall_percent",
        "either_alert_recall_percent",
    ]
    print(
        summary[display_columns].to_string(
            index=False,
            formatters={column: "{:.2f}".format for column in display_columns[2:]},
        )
    )
    return summary


def evaluate_station_false_positives(result_df: pd.DataFrame) -> pd.DataFrame:
    clean = result_df[~result_df["is_anomaly"]].copy()
    if clean.empty:
        return pd.DataFrame()

    rows = []
    for station_id, group in clean.groupby("station_id"):
        row = {"station_id": station_id, "clean_observations": len(group)}
        for method in ["statistical_alert", "spatial_alert", "both_alert", "either_alert"]:
            fp = int(group[method].sum())
            row[f"{method}_fp"] = fp
            row[f"{method}_fpr"] = fp / len(group) if len(group) else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("either_alert_fp", ascending=False)

    print("\n" + "=" * 78)
    print("FALSE POSITIVES BY STATION")
    print("=" * 78)
    print(summary.to_string(index=False, formatters={c: "{:.4f}".format for c in summary.columns if c.endswith("_fpr")}))
    return summary


def run_regional_event_sanity(
    clean_df: pd.DataFrame,
    stat_calibration,
    stat_config: StatisticalConfig,
    spatial_calibration,
    spatial_config: SpatialConfig,
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("REGIONAL EVENT SANITY TEST")
    print("=" * 78)

    test_df = clean_df.copy()
    timestamps = sorted(test_df["timestamp"].unique())
    if not timestamps:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_SEED)
    selected_timestamp = timestamps[
        rng.integers(low=len(timestamps) // 4, high=3 * len(timestamps) // 4)
    ]

    mask = test_df["timestamp"] == selected_timestamp
    affected = int(mask.sum())
    test_df.loc[mask, "temperature"] += 6.0

    stat_result = run_statistical_detector(
        test_df.copy(), VARIABLES, stat_calibration, stat_config
    )
    spatial_result = run_spatial_detector(
        test_df.copy(), VARIABLES, spatial_calibration, spatial_config
    )

    stat_rows = stat_result[stat_result["timestamp"] == selected_timestamp]
    spatial_rows = spatial_result[spatial_result["timestamp"] == selected_timestamp]

    stat_alerts = int(stat_rows["statistical_alert"].fillna(False).sum())
    spatial_alerts = int(spatial_rows["spatial_alert"].fillna(False).sum())
    any_alerts = int(
        (
            stat_rows["statistical_alert"].fillna(False).astype(bool).to_numpy()
            | spatial_rows["spatial_alert"].fillna(False).astype(bool).to_numpy()
        ).sum()
    )

    print(f"  Timestamp: {selected_timestamp}")
    print("  Variable: temperature")
    print("  Shared offset: +6.0")
    print(f"  Stations affected: {affected}")
    print(f"  Statistical alerts: {stat_alerts}/{affected}")
    print(f"  Spatial alerts:     {spatial_alerts}/{affected}")
    print(f"  Either detector:    {any_alerts}/{affected}")

    passed = spatial_alerts == 0
    print("  Spatial regional-event check: " + ("PASS" if passed else "WARNING"))

    return pd.DataFrame(
        [
            {
                "timestamp": selected_timestamp,
                "variable": "temperature",
                "regional_offset": 6.0,
                "stations_affected": affected,
                "statistical_alerts": stat_alerts,
                "spatial_alerts": spatial_alerts,
                "either_detector_alerts": any_alerts,
                "spatial_check_passed": passed,
            }
        ]
    )


def generate_plots(result_df: pd.DataFrame, metrics: pd.DataFrame, overlap: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: precision-recall-F1 comparison.
    comparison = metrics[metrics["method"].isin(["statistical", "spatial", "both_and", "either_or"])].copy()
    x = np.arange(len(comparison))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, comparison["precision"], width, label="Precision")
    ax.bar(x, comparison["recall"], width, label="Recall")
    ax.bar(x + width, comparison["f1_score"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["method"].str.replace("_", " ").str.title())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Statistical + Spatial Detector Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "detector_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: overlap counts.
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_overlap = overlap.set_index("overlap_category")["observations"]
    ax.bar(plot_overlap.index, plot_overlap.values)
    ax.set_ylabel("Observations")
    ax.set_title("Detector Overlap")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "detector_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Plots saved to: {OUTPUT_DIR}")


def save_results(
    result_df: pd.DataFrame,
    metrics: pd.DataFrame,
    overlap: pd.DataFrame,
    event_stats: pd.DataFrame,
    type_summary: pd.DataFrame,
    variable_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    regional_test: pd.DataFrame,
    injection_log: pd.DataFrame | None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "comparison_metrics.csv": metrics,
        "detector_overlap.csv": overlap,
        "event_level_results.csv": event_stats,
        "performance_by_anomaly_type.csv": type_summary,
        "performance_by_variable.csv": variable_summary,
        "false_positive_by_station.csv": station_summary,
        "regional_event_test.csv": regional_test,
    }

    if injection_log is not None and not injection_log.empty:
        injection_log.to_csv(OUTPUT_DIR / "injection_log.csv", index=False)
        print(f"✓ Saved: {OUTPUT_DIR / 'injection_log.csv'}")

    for filename, dataframe in outputs.items():
        if dataframe is not None and not dataframe.empty:
            path = OUTPUT_DIR / filename
            dataframe.to_csv(path, index=False)
            print(f"✓ Saved: {path}")

    prediction_columns = [
        "station_id",
        "timestamp",
        "is_anomaly",
        "anomaly_id",
        "anomaly_type",
        "anomaly_variable",
        "anomaly_severity",
        "statistical_alert",
        "spatial_alert",
        "both_alert",
        "either_alert",
        "statistical_only_alert",
        "spatial_only_alert",
        "neither_alert",
    ]
    available_prediction_columns = [c for c in prediction_columns if c in result_df.columns]
    predictions = result_df[available_prediction_columns].copy()
    predictions = predictions[
        predictions["is_anomaly"]
        | predictions["statistical_alert"]
        | predictions["spatial_alert"]
    ]
    predictions.to_csv(OUTPUT_DIR / "combined_anomaly_predictions.csv", index=False)
    result_df.to_parquet(OUTPUT_DIR / "combined_full_results.parquet", index=False)

    print(f"✓ Saved: {OUTPUT_DIR / 'combined_anomaly_predictions.csv'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'combined_full_results.parquet'}")


def print_recommendation(metrics: pd.DataFrame, overlap: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("HOW TO READ THE COMBINATION")
    print("=" * 78)

    lookup = metrics.set_index("method")
    for method in ["statistical", "spatial", "both_and", "either_or"]:
        row = lookup.loc[method]
        print(
            f"  {method:12s}: alerts={int(row['alert_count']):,}, "
            f"precision={row['precision']:.2%}, recall={row['recall']:.2%}, "
            f"F1={row['f1_score']:.2%}, FPR={row['false_positive_rate']:.2%}"
        )

    both = overlap.loc[overlap["overlap_category"] == "both"]
    if not both.empty:
        both_precision = float(both["precision"].iloc[0])
        print(f"\n  Agreement precision (both detectors alert): {both_precision:.2%}")

    print("\n  AND is the conservative agreement rule.")
    print("  OR is the high-recall union rule.")
    print("  The most useful next fusion step is usually an evidence-weighted rule,")
    print("  not treating the two detectors as equal voters. Use these results to decide")
    print("  whether statistical-only/spatial-only alerts should be auto-alerts or review-tier signals.")


def main() -> None:
    print("\n" + "=" * 78)
    print("SKYGUARD — STATISTICAL + SPATIAL COMBINED EVALUATOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    historical_df, clean_present_df = load_datasets()
    stat_cal, stat_cfg, spatial_cal, spatial_cfg = calibrate_detectors(historical_df)
    corrupted_df, injection_log = inject_shared_anomalies(clean_present_df)
    result_df = run_both_detectors(corrupted_df, stat_cal, stat_cfg, spatial_cal, spatial_cfg)

    metrics = evaluate_methods(result_df)
    overlap = evaluate_overlap(result_df)
    event_stats = evaluate_event_level(result_df)
    type_summary = evaluate_by_anomaly_type(result_df)
    variable_summary = evaluate_by_variable(result_df)
    station_summary = evaluate_station_false_positives(result_df)
    regional_test = run_regional_event_sanity(
        clean_present_df,
        stat_cal,
        stat_cfg,
        spatial_cal,
        spatial_cfg,
    )

    generate_plots(result_df, metrics, overlap)
    save_results(
        result_df,
        metrics,
        overlap,
        event_stats,
        type_summary,
        variable_summary,
        station_summary,
        regional_test,
        injection_log,
    )
    print_recommendation(metrics, overlap)

    print("\n" + "=" * 78)
    print("COMBINED EVALUATION COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
