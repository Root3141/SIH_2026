"""
Evaluation script for the statistical anomaly detector.

Calibrates the detector on historical weather data (2023-2025)
and evaluates its behavior on unseen real-world data (2026-present).

Note: This evaluation does not measure classification accuracy because the
2026 dataset has no ground-truth anomaly labels. Instead, it analyzes
detector behavior, alert rates, temporal patterns, and station-level patterns.
"""

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from skyguard.config.paths import (
    HISTORICAL_PARQUET,
    PRESENT_PARQUET,
    RESULTS_STATISTICAL_DIR,
)
from skyguard.detectors.statistical import (
    StatisticalConfig,
    calibrate_statistical_detector,
    run_statistical_detector,
    summarize_statistical_alerts,
)

VARIABLES = ["temperature", "humidity", "pressure"]


def load_datasets():
    """Load and validate historical and unseen evaluation datasets."""
    print("Loading datasets...")

    if not HISTORICAL_PARQUET.exists():
        raise FileNotFoundError(f"Historical dataset not found:\n{HISTORICAL_PARQUET}")

    if not PRESENT_PARQUET.exists():
        raise FileNotFoundError(f"Present dataset not found:\n{PRESENT_PARQUET}")

    historical_df = pd.read_parquet(HISTORICAL_PARQUET)
    present_df = pd.read_parquet(PRESENT_PARQUET)

    historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"], utc=True)
    present_df["timestamp"] = pd.to_datetime(present_df["timestamp"], utc=True)

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


def train_detector(historical_df, config=None):
    """Calibrate the statistical detector on historical data."""
    print("\n" + "=" * 70)
    print("CALIBRATING STATISTICAL DETECTOR")
    print("=" * 70)

    if config is None:
        config = StatisticalConfig()

    calibration = calibrate_statistical_detector(historical_df, VARIABLES, config)

    print("\n✓ Calibration complete")
    print(f"  Stations calibrated: {len(calibration.roc_thresholds)}")
    print(f"  Variables: {VARIABLES}")
    print(f"  ROC percentile: {config.roc_percentile}")
    print(f"  Z-score windows: {config.zscore_windows} hours")

    return calibration, config


def evaluate_detector(present_df, calibration, config):
    """Run the calibrated detector on unseen 2026 data."""
    print("\n" + "=" * 70)
    print("EVALUATING ON UNSEEN 2026 DATA")
    print("=" * 70)

    result_df = run_statistical_detector(present_df, VARIABLES, calibration, config)

    total_rows = len(result_df)
    alert_rows = int(result_df["statistical_alert"].sum())
    alert_rate = (alert_rows / total_rows * 100) if total_rows > 0 else 0.0
    avg_flags = (
        result_df.loc[result_df["statistical_alert"], "statistical_flag_count"].mean()
        if alert_rows > 0
        else 0.0
    )

    print("\n✓ Detection complete")
    print(f"  Total observations: {total_rows:,}")
    print(f"  Alert observations: {alert_rows:,} ({alert_rate:.4f}%)")
    print(f"  Average flags per alert: {avg_flags:.2f}")

    return result_df


def generate_alert_summary(result_df):
    """Generate alert counts and rates by detector."""
    print("\n" + "=" * 70)
    print("ALERT SUMMARY BY DETECTOR")
    print("=" * 70)

    summary = summarize_statistical_alerts(result_df, VARIABLES)
    summary["alert_rate_percent"] = summary["alert_rate"] * 100
    summary = summary.sort_values("alert_count", ascending=False)

    display_columns = ["variable", "detector", "alert_count", "alert_rate_percent"]
    print(summary[display_columns].to_string(index=False))

    return summary


def generate_station_statistics(result_df):
    """Generate alert statistics for each station."""
    print("\n" + "=" * 70)
    print("ALERTS BY STATION")
    print("=" * 70)

    station_stats = result_df.groupby("station_id").agg(
        alert_count=("statistical_alert", "sum"),
        alert_rate=("statistical_alert", "mean"),
        avg_flags=("statistical_flag_count", "mean"),
        max_flags=("statistical_flag_count", "max"),
        total_observations=("timestamp", "count"),
    )

    station_stats["alert_rate_percent"] = station_stats["alert_rate"] * 100
    station_stats = station_stats.sort_values("alert_rate_percent", ascending=False)

    display_columns = [
        "alert_count",
        "alert_rate_percent",
        "avg_flags",
        "max_flags",
        "total_observations",
    ]
    print(station_stats[display_columns].round(4).to_string())

    return station_stats


def generate_temporal_analysis(result_df):
    """Analyze alert patterns across dates and hours."""
    print("\n" + "=" * 70)
    print("TEMPORAL ALERT PATTERNS")
    print("=" * 70)

    df = result_df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    daily_alerts = df.groupby("date").agg(
        alert_count=("statistical_alert", "sum"),
        alert_rate=("statistical_alert", "mean"),
    )
    daily_alerts["alert_rate_percent"] = daily_alerts["alert_rate"] * 100

    hourly_alerts = df.groupby("hour").agg(
        alert_count=("statistical_alert", "sum"),
        alert_rate=("statistical_alert", "mean"),
    )
    hourly_alerts["alert_rate_percent"] = hourly_alerts["alert_rate"] * 100

    print("\nTop 10 days by alert count:")
    print(
        daily_alerts.sort_values("alert_count", ascending=False)
        .head(10)
        .round(4)
        .to_string()
    )

    print("\nHourly alert patterns:")
    print(hourly_alerts.round(4).to_string())

    return daily_alerts, hourly_alerts


def show_alert_examples(result_df, n=10):
    """Display sample alerts for manual inspection."""
    print("\n" + "=" * 70)
    print("SAMPLE ALERTS")
    print("=" * 70)

    alerts = result_df[result_df["statistical_alert"]].copy()

    if alerts.empty:
        print("No alerts detected.")
        return

    anomaly_columns = [col for col in alerts.columns if col.endswith("_anomaly")]
    base_columns = [
        "timestamp",
        "station_id",
        "temperature",
        "humidity",
        "pressure",
        "statistical_flag_count",
    ]
    available_columns = [col for col in base_columns if col in alerts.columns]
    display_columns = available_columns + anomaly_columns

    print(alerts[display_columns].head(n).to_string(index=False))


def plot_detector_evaluation(result_df, output_dir):
    """Generate visualizations describing detector behavior on unseen data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = result_df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Statistical Detector Behavior on Unseen 2026 Data",
        fontsize=16,
        fontweight="bold",
    )

    daily_alert_rate = df.groupby("date")["statistical_alert"].mean() * 100
    axes[0, 0].plot(daily_alert_rate.index, daily_alert_rate.values, linewidth=1.2)
    axes[0, 0].set_title("Daily Alert Rate")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Alert Rate (%)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=45)

    hourly_alert_rate = df.groupby("hour")["statistical_alert"].mean() * 100
    axes[0, 1].bar(hourly_alert_rate.index, hourly_alert_rate.values)
    axes[0, 1].set_title("Hourly Alert Rate Pattern")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Alert Rate (%)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    station_alert_rate = df.groupby("station_id")["statistical_alert"].mean() * 100
    station_alert_rate = station_alert_rate.sort_values(ascending=True)
    axes[1, 0].barh(station_alert_rate.index, station_alert_rate.values)
    axes[1, 0].set_title("Alert Rate by Station")
    axes[1, 0].set_xlabel("Alert Rate (%)")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    alerts = df[df["statistical_alert"]]
    if not alerts.empty:
        flag_dist = alerts["statistical_flag_count"].value_counts().sort_index()
        axes[1, 1].bar(flag_dist.index, flag_dist.values)
        axes[1, 1].set_xlabel("Number of Triggered Detectors")
        axes[1, 1].set_ylabel("Number of Observations")
    else:
        axes[1, 1].text(
            0.5, 0.5, "No alerts detected", ha="center", va="center", fontsize=14
        )
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    axes[1, 1].set_title("Flag Count Distribution")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "detector_evaluation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Plot saved:\n  {output_path}")


def save_results(
    result_df, summary_df, station_stats, daily_alerts, hourly_alerts, output_dir
):
    """Save detailed evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    alert_df = result_df[result_df["statistical_alert"]].copy().sort_values("timestamp")
    alerts_path = output_dir / "alerts.csv"
    alert_df.to_csv(alerts_path, index=False)
    print(f"✓ Alerts saved: {alerts_path} ({len(alert_df):,} rows)")

    summary_path = output_dir / "alert_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Alert summary saved:\n  {summary_path}")

    station_path = output_dir / "station_statistics.csv"
    station_stats.to_csv(station_path)
    print(f"✓ Station statistics saved:\n  {station_path}")

    daily_path = output_dir / "daily_alert_statistics.csv"
    daily_alerts.to_csv(daily_path)
    print(f"✓ Daily alert statistics saved:\n  {daily_path}")

    hourly_path = output_dir / "hourly_alert_statistics.csv"
    hourly_alerts.to_csv(hourly_path)
    print(f"✓ Hourly alert statistics saved:\n  {hourly_path}")

    full_results_path = output_dir / "full_results.parquet"
    result_df.to_parquet(full_results_path, index=False)
    print(f"✓ Full results saved:\n  {full_results_path}")


def main():
    print("\n" + "=" * 70)
    print("STATISTICAL DETECTOR EVALUATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    historical_df, present_df = load_datasets()

    calibration, config = train_detector(historical_df)

    result_df = evaluate_detector(present_df, calibration, config)

    summary_df = generate_alert_summary(result_df)

    station_stats = generate_station_statistics(result_df)

    daily_alerts, hourly_alerts = generate_temporal_analysis(result_df)

    show_alert_examples(result_df, n=10)

    plot_detector_evaluation(result_df, RESULTS_STATISTICAL_DIR)

    save_results(
        result_df, summary_df, station_stats, daily_alerts, hourly_alerts, RESULTS_STATISTICAL_DIR
    )

    print("\n" + "=" * 70)
    print(f"Evaluation complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to:\n{RESULTS_STATISTICAL_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
