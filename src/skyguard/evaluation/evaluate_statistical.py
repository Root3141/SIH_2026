"""
Evaluate the statistical anomaly detector.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyguard.config.paths import (
    HISTORICAL_PARQUET,
    PRESENT_PARQUET,
    RESULTS_STATISTICAL_DIR,
    SYNTHETIC_DATA_DIR,
)
from skyguard.detectors.statistical import (
    StatisticalConfig,
    calibrate_statistical_detector,
    run_statistical_detector,
    summarize_statistical_alerts,
)
from skyguard.simulation.anomaly_injector import (
    AnomalyConfig,
    inject_anomalies,
)

VARIABLES = ["temperature", "humidity", "pressure"]
RANDOM_SEED = 42
STATISTICAL_INJECTED_PARQUET = (
    SYNTHETIC_DATA_DIR / "statistical_evaluation_injected.parquet"
)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load calibration and evaluation datasets."""

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    if not HISTORICAL_PARQUET.exists():
        raise FileNotFoundError(f"Historical dataset not found:\n{HISTORICAL_PARQUET}")

    if not PRESENT_PARQUET.exists():
        raise FileNotFoundError(f"Present dataset not found:\n{PRESENT_PARQUET}")

    historical_df = pd.read_parquet(HISTORICAL_PARQUET)
    present_df = pd.read_parquet(PRESENT_PARQUET)

    for df in (historical_df, present_df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    historical_df = historical_df.sort_values(["station_id", "timestamp"]).reset_index(
        drop=True
    )
    present_df = present_df.sort_values(["station_id", "timestamp"]).reset_index(
        drop=True
    )

    print("\nHistorical calibration data:")
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


def calibrate_detector(historical_df: pd.DataFrame) -> tuple[object, StatisticalConfig]:
    """Calibrate the detector using historical data only."""

    print("\n" + "=" * 70)
    print("CALIBRATING STATISTICAL DETECTOR")
    print("=" * 70)

    config = StatisticalConfig()
    calibration = calibrate_statistical_detector(historical_df, VARIABLES, config)

    print("\n✓ Calibration complete")
    print(f"  Stations calibrated: {len(calibration.roc_thresholds)}")
    print(f"  Variables: {VARIABLES}")
    print(f"  ROC percentile: {config.roc_percentile}")
    print(f"  Z-score threshold: {config.zscore_threshold}")
    print(f"  Z-score windows: {config.zscore_windows}")
    print(f"  Persistence window: {config.persistence_window}")

    return calibration, config


def inject_evaluation_anomalies(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject synthetic anomalies into unseen evaluation data."""
    print("\n" + "=" * 70)
    print("INJECTING SYNTHETIC ANOMALIES")
    print("=" * 70)

    config = AnomalyConfig(anomaly_rate=0.02, random_seed=RANDOM_SEED)
    corrupted_df, injection_log = inject_anomalies(clean_df, VARIABLES, config)
    corrupted_df = corrupted_df.copy()

    if "is_anomaly" not in corrupted_df.columns:
        corrupted_df["is_anomaly"] = (
            corrupted_df.get(
                "synthetic_anomaly", pd.Series(False, index=corrupted_df.index)
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

    required_ground_truth = {
        "is_anomaly",
        "anomaly_id",
        "anomaly_type",
        "anomaly_variable",
        "anomaly_severity",
    }
    missing = required_ground_truth - set(corrupted_df.columns)

    if missing:
        raise ValueError(
            f"Missing ground-truth columns after anomaly injection: {sorted(missing)}"
        )

    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corrupted_df.to_parquet(STATISTICAL_INJECTED_PARQUET, index=False)

    anomaly_count = int(corrupted_df["is_anomaly"].fillna(False).sum())
    anomaly_rate = (
        anomaly_count / len(corrupted_df) * 100 if len(corrupted_df) > 0 else 0.0
    )

    print("\n✓ Anomaly injection complete")
    print(f"  Total observations:      {len(corrupted_df):,}")
    print(f"  Anomalous observations:  {anomaly_count:,}")
    print(f"  Actual anomaly rate:     {anomaly_rate:.3f}%")

    if injection_log is not None:
        print(f"  Injected anomaly events: {len(injection_log):,}")

    print(f"\n✓ Injected dataset saved:\n  {STATISTICAL_INJECTED_PARQUET}")

    return corrupted_df, injection_log


def run_detector(
    evaluation_df: pd.DataFrame, calibration, config: StatisticalConfig
) -> pd.DataFrame:
    """Run the calibrated statistical detector."""
    print("\n" + "=" * 70)
    print("RUNNING STATISTICAL DETECTOR")
    print("=" * 70)

    result_df = run_statistical_detector(evaluation_df, VARIABLES, calibration, config)

    total_rows = len(result_df)
    alert_count = int(result_df["statistical_alert"].fillna(False).sum())
    alert_rate = alert_count / total_rows * 100 if total_rows > 0 else 0.0

    print("\n✓ Detection complete")
    print(f"  Total observations: {total_rows:,}")
    print(f"  Statistical alerts: {alert_count:,}")
    print(f"  Alert rate:         {alert_rate:.4f}%")

    return result_df


def calculate_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate binary classification metrics."""
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
    """Evaluate observation-level anomaly detection performance."""
    print("\n" + "=" * 70)
    print("OBSERVATION-LEVEL PERFORMANCE")
    print("=" * 70)

    if "is_anomaly" not in result_df.columns:
        raise ValueError("Ground-truth column 'is_anomaly' not found.")

    metrics = calculate_binary_metrics(
        result_df["is_anomaly"], result_df["statistical_alert"]
    )
    metrics_df = pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in metrics.items()]
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
    """Evaluate event-level detection."""
    print("\n" + "=" * 70)
    print("EVENT-LEVEL PERFORMANCE")
    print("=" * 70)

    anomaly_rows = result_df[result_df["is_anomaly"].fillna(False)].copy()

    if anomaly_rows.empty:
        print("No anomaly observations found.")
        return pd.DataFrame()

    if "anomaly_id" not in anomaly_rows.columns:
        print("⚠ anomaly_id not found. Skipping event evaluation.")
        return pd.DataFrame()

    event_stats = (
        anomaly_rows.dropna(subset=["anomaly_id"])
        .groupby("anomaly_id")
        .agg(
            anomaly_type=("anomaly_type", "first"),
            variable=("anomaly_variable", "first"),
            severity=("anomaly_severity", "first"),
            duration=("anomaly_id", "size"),
            detected=("statistical_alert", "any"),
            observations_detected=("statistical_alert", "sum"),
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
    """Evaluate recall by synthetic anomaly type."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BY ANOMALY TYPE")
    print("=" * 70)

    anomaly_rows = result_df[result_df["is_anomaly"].fillna(False)].copy()

    if anomaly_rows.empty:
        return pd.DataFrame()

    rows = []

    for anomaly_type, group in anomaly_rows.groupby("anomaly_type", dropna=False):
        total = len(group)
        detected = int(group["statistical_alert"].fillna(False).sum())
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
    """Evaluate recall by injected anomaly variable."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BY ANOMALY VARIABLE")
    print("=" * 70)

    anomaly_rows = result_df[result_df["is_anomaly"].fillna(False)].copy()
    rows = []

    for variable in VARIABLES:
        variable_rows = anomaly_rows[anomaly_rows["anomaly_variable"] == variable]
        total = len(variable_rows)
        detected = int(variable_rows["statistical_alert"].fillna(False).sum())
        recall = detected / total if total > 0 else 0.0

        rows.append(
            {
                "variable": variable,
                "anomaly_observations": total,
                "detected": detected,
                "missed": total - detected,
                "recall": recall,
                "recall_percent": recall * 100,
            }
        )

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False,
            formatters={"recall": "{:.4f}".format, "recall_percent": "{:.2f}".format},
        )
    )

    return summary


def analyze_false_positives(result_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze false positives by station."""
    print("\n" + "=" * 70)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 70)

    clean_rows = result_df[~result_df["is_anomaly"].fillna(False)].copy()
    false_positives = clean_rows[clean_rows["statistical_alert"].fillna(False)].copy()

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


def get_detector_columns(result_df: pd.DataFrame) -> list[str]:
    """Return real statistical detector flag columns."""
    columns = []

    for variable in VARIABLES:
        expected = [
            f"{variable}_range_anomaly",
            f"{variable}_roc_anomaly",
            f"{variable}_persistence_anomaly",
        ]
        columns.extend(column for column in expected if column in result_df.columns)
        columns.extend(
            column
            for column in result_df.columns
            if column.startswith(f"{variable}_zscore_") and column.endswith("_anomaly")
        )

    return sorted(set(columns))


def analyze_detector_firing_rates(result_df: pd.DataFrame) -> pd.DataFrame:
    """Show how often individual statistical signals fire."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL DETECTOR FIRING RATES")
    print("=" * 70)

    detector_columns = get_detector_columns(result_df)
    rows = []

    for column in detector_columns:
        count = int(result_df[column].fillna(False).astype(bool).sum())
        rate = count / len(result_df) if len(result_df) > 0 else 0.0
        rows.append(
            {
                "detector": column,
                "alert_count": count,
                "alert_rate": rate,
                "alert_rate_percent": rate * 100,
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values("alert_count", ascending=False)
        .reset_index(drop=True)
    )

    if not summary.empty:
        print(
            summary[["detector", "alert_count", "alert_rate_percent"]]
            .round(4)
            .to_string(index=False)
        )

    return summary


def analyze_evidence_families(result_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze detector evidence families."""
    print("\n" + "=" * 70)
    print("EVIDENCE FAMILY ANALYSIS")
    print("=" * 70)

    family_columns = [
        "evidence_range",
        "evidence_roc",
        "evidence_level",
        "evidence_persistence",
    ]
    available_columns = [
        column for column in family_columns if column in result_df.columns
    ]

    if not available_columns:
        print("No evidence-family columns found.")
        return pd.DataFrame()

    rows = []
    for column in available_columns:
        count = int(result_df[column].fillna(False).astype(bool).sum())
        rate = count / len(result_df) if len(result_df) > 0 else 0.0
        rows.append(
            {
                "evidence_family": column.replace("evidence_", ""),
                "observations": count,
                "rate": rate,
                "rate_percent": rate * 100,
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values("observations", ascending=False)
        .reset_index(drop=True)
    )
    print(summary.round(4).to_string(index=False))

    return summary


def analyze_station_alerts(result_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze alert rates by station."""
    print("\n" + "=" * 70)
    print("ALERT RATE BY STATION")
    print("=" * 70)

    station_summary = (
        result_df.groupby("station_id")
        .agg(
            total_observations=("timestamp", "size"),
            alert_count=("statistical_alert", "sum"),
            alert_rate=("statistical_alert", "mean"),
        )
        .reset_index()
    )
    station_summary["alert_rate_percent"] = station_summary["alert_rate"] * 100
    station_summary = station_summary.sort_values("alert_rate", ascending=False)

    print(
        station_summary[
            ["station_id", "total_observations", "alert_count", "alert_rate_percent"]
        ]
        .round(4)
        .to_string(index=False)
    )

    return station_summary


def analyze_severity_distribution(result_df: pd.DataFrame) -> pd.DataFrame:
    """Inspect anomaly severity distribution."""
    print("\n" + "=" * 70)
    print("STATISTICAL SEVERITY DISTRIBUTION")
    print("=" * 70)

    severity_column = "statistical_severity_label"

    if severity_column not in result_df.columns:
        print("Severity column not found.")
        return pd.DataFrame()

    summary = (
        result_df.groupby(severity_column)
        .agg(
            observations=("station_id", "size"),
            injected_anomalies=("is_anomaly", "sum"),
            alerts=("statistical_alert", "sum"),
        )
        .reset_index()
    )
    summary["anomaly_rate_percent"] = (
        summary["injected_anomalies"] / summary["observations"] * 100
    )

    print(summary.round(4).to_string(index=False))
    return summary


def generate_alert_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """Generate alert counts for individual detector signals."""
    print("\n" + "=" * 70)
    print("ALERT SUMMARY")
    print("=" * 70)

    summary = summarize_statistical_alerts(result_df, VARIABLES)
    summary = summary[
        ~summary["detector"].str.contains("synthetic", case=False, na=False)
    ].copy()
    summary["alert_rate_percent"] = summary["alert_rate"] * 100
    summary = summary.sort_values("alert_count", ascending=False)

    print(
        summary[["variable", "detector", "alert_count", "alert_rate_percent"]]
        .round(4)
        .to_string(index=False)
    )

    return summary


def generate_plots(
    result_df: pd.DataFrame, anomaly_type_summary: pd.DataFrame, output_dir: Path
) -> None:
    """Generate compact evaluation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statistical Detector Evaluation", fontsize=16, fontweight="bold")

    y_true = result_df["is_anomaly"].fillna(False).astype(bool)
    y_pred = result_df["statistical_alert"].fillna(False).astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    matrix = np.array([[tn, fp], [fn, tp]])
    axes[0, 0].imshow(matrix)
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

    if anomaly_type_summary is not None and not anomaly_type_summary.empty:
        axes[0, 1].bar(
            anomaly_type_summary["anomaly_type"].astype(str),
            anomaly_type_summary["recall_percent"],
        )
        axes[0, 1].set_title("Recall by Anomaly Type")
        axes[0, 1].set_xlabel("Anomaly Type")
        axes[0, 1].set_ylabel("Recall (%)")
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(axis="y", alpha=0.3)

    station_alert_rate = (
        result_df.groupby("station_id")["statistical_alert"].mean().sort_values() * 100
    )
    axes[1, 0].barh(station_alert_rate.index.astype(str), station_alert_rate.values)
    axes[1, 0].set_title("Alert Rate by Station")
    axes[1, 0].set_xlabel("Alert Rate (%)")
    axes[1, 0].grid(axis="x", alpha=0.3)

    if "statistical_severity_label" in result_df.columns:
        severity_counts = result_df["statistical_severity_label"].value_counts()
        axes[1, 1].bar(severity_counts.index.astype(str), severity_counts.values)
        axes[1, 1].set_title("Statistical Severity Distribution")
        axes[1, 1].set_xlabel("Severity")
        axes[1, 1].set_ylabel("Observations")
        axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "statistical_detector_evaluation.png"
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
    detector_summary: pd.DataFrame,
    evidence_family_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    severity_summary: pd.DataFrame,
    injection_log: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Save evaluation outputs."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    if injection_log is not None and not injection_log.empty:
        path = output_dir / "injection_log.csv"
        injection_log.to_csv(path, index=False)
        print(f"✓ Saved: {path}")

    outputs = {
        "observation_metrics.csv": observation_metrics,
        "event_metrics.csv": event_metrics,
        "performance_by_anomaly_type.csv": anomaly_type_summary,
        "performance_by_variable.csv": variable_summary,
        "false_positive_analysis.csv": false_positive_summary,
        "detector_firing_rates.csv": detector_summary,
        "evidence_family_summary.csv": evidence_family_summary,
        "station_statistics.csv": station_summary,
        "severity_distribution.csv": severity_summary,
    }

    for filename, dataframe in outputs.items():
        if dataframe is not None and not dataframe.empty:
            path = output_dir / filename
            dataframe.to_csv(path, index=False)
            print(f"✓ Saved: {path}")

    anomaly_predictions = result_df[
        result_df["is_anomaly"].fillna(False)
        | result_df["statistical_alert"].fillna(False)
    ].copy()

    path = output_dir / "anomaly_predictions.csv"
    anomaly_predictions.to_csv(path, index=False)
    print(f"✓ Saved: {path} ({len(anomaly_predictions):,} rows)")

    path = output_dir / "full_results.parquet"
    result_df.to_parquet(path, index=False)
    print(f"✓ Saved: {path}")


def main() -> None:
    print("\n" + "=" * 70)
    print("STATISTICAL DETECTOR EVALUATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    historical_df, clean_present_df = load_datasets()
    calibration, config = calibrate_detector(historical_df)
    corrupted_df, injection_log = inject_evaluation_anomalies(clean_present_df)
    result_df = run_detector(corrupted_df, calibration, config)

    observation_metrics = evaluate_observation_level(result_df)
    event_metrics = evaluate_event_level(result_df)
    anomaly_type_summary = evaluate_by_anomaly_type(result_df)
    variable_summary = evaluate_by_variable(result_df)
    false_positive_summary = analyze_false_positives(result_df)

    detector_summary = analyze_detector_firing_rates(result_df)
    evidence_family_summary = analyze_evidence_families(result_df)
    station_summary = analyze_station_alerts(result_df)
    severity_summary = analyze_severity_distribution(result_df)

    generate_plots(result_df, anomaly_type_summary, RESULTS_STATISTICAL_DIR)

    save_results(
        result_df=result_df,
        observation_metrics=observation_metrics,
        event_metrics=event_metrics,
        anomaly_type_summary=anomaly_type_summary,
        variable_summary=variable_summary,
        false_positive_summary=false_positive_summary,
        detector_summary=detector_summary,
        evidence_family_summary=evidence_family_summary,
        station_summary=station_summary,
        severity_summary=severity_summary,
        injection_log=injection_log,
        output_dir=RESULTS_STATISTICAL_DIR,
    )

    print("\n" + "=" * 70)
    print("STATISTICAL EVALUATION COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to:\n{RESULTS_STATISTICAL_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
