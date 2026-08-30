from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

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

    historical_df = historical_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    present_df = present_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print("\nHistorical calibration data:")
    print(f"  Rows:     {len(historical_df):,}")
    print(f"  Stations: {historical_df['station_id'].nunique()}")
    print(f"  Period:   {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")

    print("\nUnseen evaluation data:")
    print(f"  Rows:     {len(present_df):,}")
    print(f"  Stations: {present_df['station_id'].nunique()}")
    print(f"  Period:   {present_df['timestamp'].min()} to {present_df['timestamp'].max()}")

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
    print(f"  Z-score threshold: {config.zscore_threshold}")
    print(f"  Z-score windows: {config.zscore_windows} hours")
    print(f"  Persistence window: {config.persistence_window} hours")

    return calibration, config


def evaluate_detector(present_df, calibration, config):
    """Run the calibrated detector on unseen 2026 data."""
    print("\n" + "=" * 70)
    print("EVALUATING ON UNSEEN 2026 DATA")
    print("=" * 70)

    result_df = run_statistical_detector(present_df, VARIABLES, calibration, config)

    total_rows = len(result_df)
    alert_rows = int(result_df["statistical_alert"].sum())
    alert_rate = alert_rows / total_rows * 100 if total_rows > 0 else 0.0
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
    """Generate alert counts and rates by individual detector."""
    print("\n" + "=" * 70)
    print("ALERT SUMMARY BY INDIVIDUAL DETECTOR")
    print("=" * 70)

    summary = summarize_statistical_alerts(result_df, VARIABLES)
    summary["alert_rate_percent"] = summary["alert_rate"] * 100
    summary = summary.sort_values("alert_count", ascending=False)

    display_columns = ["variable", "detector", "alert_count", "alert_rate_percent"]

    print(summary[display_columns].round(4).to_string(index=False))
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
    print(daily_alerts.sort_values("alert_count", ascending=False).head(10).round(4).to_string())

    print("\nHourly alert patterns:")
    print(hourly_alerts.round(4).to_string())

    return daily_alerts, hourly_alerts


def analyze_flag_count_distribution(result_df):
    """Analyze how many detector signals fire simultaneously."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: FLAG COUNT DISTRIBUTION")
    print("=" * 70)

    total_rows = len(result_df)
    distribution = (
        result_df["statistical_flag_count"]
        .value_counts()
        .sort_index()
        .rename_axis("flag_count")
        .reset_index(name="observation_count")
    )
    distribution["observation_rate"] = distribution["observation_count"] / total_rows
    distribution["observation_rate_percent"] = distribution["observation_rate"] * 100

    print(
        distribution[["flag_count", "observation_count", "observation_rate_percent"]]
        .round(4)
        .to_string(index=False)
    )

    alert_distribution = distribution[distribution["flag_count"] > 0].copy()
    total_alerts = alert_distribution["observation_count"].sum()

    if total_alerts > 0:
        alert_distribution["share_of_alerts_percent"] = (
            alert_distribution["observation_count"] / total_alerts * 100
        )
        print("\nAmong alert observations:")
        print(
            alert_distribution[["flag_count", "observation_count", "share_of_alerts_percent"]]
            .round(4)
            .to_string(index=False)
        )

    return distribution


def add_evidence_family_columns(result_df):
    """Add family-level detector booleans."""
    df = result_df.copy()

    range_columns = [
        f"{variable}_range_anomaly"
        for variable in VARIABLES
        if f"{variable}_range_anomaly" in df.columns
    ]
    roc_columns = [
        f"{variable}_roc_anomaly"
        for variable in VARIABLES
        if f"{variable}_roc_anomaly" in df.columns
    ]
    persistence_columns = [
        f"{variable}_persistence_anomaly"
        for variable in VARIABLES
        if f"{variable}_persistence_anomaly" in df.columns
    ]

    level_columns = []
    for variable in VARIABLES:
        for column in df.columns:
            if column.startswith(f"{variable}_zscore_") and column.endswith("_anomaly"):
                level_columns.append(column)

    def any_flag(columns):
        if not columns:
            return pd.Series(False, index=df.index, dtype=bool)
        return df[columns].fillna(False).astype(bool).any(axis=1)

    df["family_range"] = any_flag(range_columns)
    df["family_roc"] = any_flag(roc_columns)
    df["family_level"] = any_flag(level_columns)
    df["family_persistence"] = any_flag(persistence_columns)

    family_columns = ["family_range", "family_roc", "family_level", "family_persistence"]
    df["evidence_family_count"] = df[family_columns].sum(axis=1)

    return df


def analyze_evidence_families(result_df):
    """Analyze firing rates of conceptual evidence families."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: EVIDENCE FAMILY ANALYSIS")
    print("=" * 70)

    df = add_evidence_family_columns(result_df)

    family_columns = ["family_range", "family_roc", "family_level", "family_persistence"]
    rows = []
    for column in family_columns:
        count = int(df[column].sum())
        rate = count / len(df) * 100 if len(df) > 0 else 0.0
        rows.append(
            {
                "evidence_family": column.replace("family_", ""),
                "observation_count": count,
                "observation_rate_percent": rate,
            }
        )

    family_summary = pd.DataFrame(rows)
    print(family_summary.sort_values("observation_count", ascending=False).round(4).to_string(index=False))

    return df, family_summary


def analyze_evidence_combinations(df):
    """Analyze combinations of evidence families."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: EVIDENCE FAMILY COMBINATIONS")
    print("=" * 70)

    family_mapping = {
        "family_range": "RANGE",
        "family_roc": "ROC",
        "family_level": "LEVEL",
        "family_persistence": "PERSISTENCE",
    }
    family_columns = list(family_mapping.keys())

    def build_combination(row):
        active_families = [family_mapping[column] for column in family_columns if bool(row[column])]
        return " + ".join(active_families) if active_families else "NONE"

    df = df.copy()
    df["evidence_combination"] = df.apply(build_combination, axis=1)

    combination_summary = (
        df["evidence_combination"]
        .value_counts()
        .rename_axis("evidence_combination")
        .reset_index(name="observation_count")
    )
    combination_summary["observation_rate_percent"] = (
        combination_summary["observation_count"] / len(df) * 100
    )

    alert_combinations = combination_summary[combination_summary["evidence_combination"] != "NONE"].copy()
    total_alert_family_rows = alert_combinations["observation_count"].sum()

    if total_alert_family_rows > 0:
        alert_combinations["share_of_family_alerts_percent"] = (
            alert_combinations["observation_count"] / total_alert_family_rows * 100
        )

    print(alert_combinations.round(4).to_string(index=False))
    return combination_summary


def analyze_variable_involvement(result_df):
    """Analyze which weather variables contribute evidence."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: VARIABLE INVOLVEMENT")
    print("=" * 70)

    df = result_df.copy()

    for variable in VARIABLES:
        columns = [
            column
            for column in df.columns
            if (column.startswith(f"{variable}_") and column.endswith("_anomaly"))
        ]
        if columns:
            df[f"{variable}_involved"] = df[columns].fillna(False).astype(bool).any(axis=1)
        else:
            df[f"{variable}_involved"] = False

    def build_variable_combination(row):
        active_variables = [
            variable.upper() for variable in VARIABLES if bool(row[f"{variable}_involved"])
        ]
        return " + ".join(active_variables) if active_variables else "NONE"

    df["variable_combination"] = df.apply(build_variable_combination, axis=1)

    summary = (
        df["variable_combination"]
        .value_counts()
        .rename_axis("variable_combination")
        .reset_index(name="observation_count")
    )
    summary["observation_rate_percent"] = summary["observation_count"] / len(df) * 100

    alert_summary = summary[summary["variable_combination"] != "NONE"].copy()
    total_alerts = alert_summary["observation_count"].sum()

    if total_alerts > 0:
        alert_summary["share_of_alerts_percent"] = (
            alert_summary["observation_count"] / total_alerts * 100
        )

    print(alert_summary.round(4).to_string(index=False))
    return summary


def analyze_alert_streaks(result_df):
    """Analyze consecutive alert streaks separately for each station."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: ALERT STREAK ANALYSIS")
    print("=" * 70)

    df = result_df.copy().sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    streak_records = []

    for station_id, station_df in df.groupby("station_id"):
        station_df = station_df.copy()
        alert = station_df["statistical_alert"].astype(bool)
        previous_alert = alert.shift(fill_value=False)
        timestamp_gap = station_df["timestamp"].diff().dt.total_seconds().div(3600)
        new_streak = alert & (~previous_alert | (timestamp_gap != 1))

        station_df["streak_id"] = new_streak.cumsum()
        alert_rows = station_df[alert]

        for streak_id, streak_df in alert_rows.groupby("streak_id"):
            if streak_df.empty:
                continue

            start_time = streak_df["timestamp"].min()
            end_time = streak_df["timestamp"].max()
            duration_hours = int((end_time - start_time).total_seconds() / 3600) + 1

            streak_records.append(
                {
                    "station_id": station_id,
                    "streak_id": int(streak_id),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_hours": duration_hours,
                    "observations": len(streak_df),
                    "max_flag_count": int(streak_df["statistical_flag_count"].max()),
                    "avg_flag_count": float(streak_df["statistical_flag_count"].mean()),
                }
            )

    streak_df = pd.DataFrame(streak_records)

    if streak_df.empty:
        print("No alert streaks detected.")
        return streak_df, pd.DataFrame()

    streak_distribution = (
        streak_df["duration_hours"]
        .value_counts()
        .sort_index()
        .rename_axis("duration_hours")
        .reset_index(name="streak_count")
    )
    streak_distribution["share_percent"] = streak_distribution["streak_count"] / len(streak_df) * 100

    print(f"Total alert streaks: {len(streak_df):,}")
    print("\nStreak duration distribution:")
    print(streak_distribution.round(4).to_string(index=False))
    print("\nLongest alert streaks:")
    print(streak_df.sort_values("duration_hours", ascending=False).head(10).to_string(index=False))

    return streak_df, streak_distribution


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


def plot_detector_evaluation(result_df, family_df, output_dir):
    """Generate evaluation and diagnostic visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = result_df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statistical Detector Behavior on Unseen 2026 Data", fontsize=16, fontweight="bold")

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

    flag_dist = df["statistical_flag_count"].value_counts().sort_index()
    axes[1, 1].bar(flag_dist.index, flag_dist.values)
    axes[1, 1].set_title("Distribution of Detector Flag Counts")
    axes[1, 1].set_xlabel("Number of Triggered Detectors")
    axes[1, 1].set_ylabel("Number of Observations")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "detector_evaluation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Evaluation plot saved:\n  {output_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statistical Detector Diagnostic Analysis", fontsize=16, fontweight="bold")

    anomaly_columns = [col for col in df.columns if col.endswith("_anomaly")]
    detector_rates = df[anomaly_columns].fillna(False).astype(bool).mean() * 100
    detector_rates = detector_rates.sort_values(ascending=True)
    axes[0, 0].barh(detector_rates.index, detector_rates.values)
    axes[0, 0].set_title("Individual Detector Firing Rates")
    axes[0, 0].set_xlabel("Observation Rate (%)")
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    family_columns = ["family_range", "family_roc", "family_level", "family_persistence"]
    family_rates = family_df[family_columns].mean() * 100
    family_rates.index = [column.replace("family_", "").title() for column in family_rates.index]
    axes[0, 1].bar(family_rates.index, family_rates.values)
    axes[0, 1].set_title("Evidence Family Firing Rates")
    axes[0, 1].set_ylabel("Observation Rate (%)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    family_count_dist = family_df["evidence_family_count"].value_counts().sort_index()
    axes[1, 0].bar(family_count_dist.index, family_count_dist.values)
    axes[1, 0].set_title("Independent Evidence Families per Observation")
    axes[1, 0].set_xlabel("Number of Evidence Families")
    axes[1, 0].set_ylabel("Number of Observations")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    variable_rates = {}
    for variable in VARIABLES:
        anomaly_columns = [
            column for column in df.columns if (column.startswith(f"{variable}_") and column.endswith("_anomaly"))
        ]
        if anomaly_columns:
            variable_rates[variable] = (
                df[anomaly_columns].fillna(False).astype(bool).any(axis=1).mean() * 100
            )

    axes[1, 1].bar(list(variable_rates.keys()), list(variable_rates.values()))
    axes[1, 1].set_title("Variable Involvement Rate")
    axes[1, 1].set_ylabel("Observation Rate (%)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    diagnostic_path = output_dir / "diagnostic_analysis.png"
    plt.savefig(diagnostic_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Diagnostic plot saved:\n  {diagnostic_path}")


def save_results(
    result_df,
    summary_df,
    station_stats,
    daily_alerts,
    hourly_alerts,
    flag_distribution,
    family_summary,
    combination_summary,
    variable_summary,
    streak_df,
    streak_distribution,
    output_dir
):
    """Save all evaluation and diagnostic outputs."""
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

    flag_distribution_path = output_dir / "flag_count_distribution.csv"
    flag_distribution.to_csv(flag_distribution_path, index=False)
    print(f"✓ Flag-count distribution saved:\n  {flag_distribution_path}")

    family_summary_path = output_dir / "evidence_family_summary.csv"
    family_summary.to_csv(family_summary_path, index=False)
    print(f"✓ Evidence-family summary saved:\n  {family_summary_path}")

    combination_path = output_dir / "evidence_combinations.csv"
    combination_summary.to_csv(combination_path, index=False)
    print(f"✓ Evidence combinations saved:\n  {combination_path}")

    variable_path = output_dir / "variable_involvement.csv"
    variable_summary.to_csv(variable_path, index=False)
    print(f"✓ Variable involvement saved:\n  {variable_path}")

    if not streak_df.empty:
        streak_path = output_dir / "alert_streaks.csv"
        streak_df.to_csv(streak_path, index=False)
        print(f"✓ Alert streaks saved:\n  {streak_path}")

    if not streak_distribution.empty:
        streak_distribution_path = output_dir / "alert_streak_distribution.csv"
        streak_distribution.to_csv(streak_distribution_path, index=False)
        print(f"✓ Alert streak distribution saved:\n  {streak_distribution_path}")

    full_results_path = output_dir / "full_results.parquet"
    result_df.to_parquet(full_results_path, index=False)
    print(f"✓ Full results saved:\n  {full_results_path}")


def main():
    print("\n" + "=" * 70)
    print("STATISTICAL DETECTOR EVALUATION + DIAGNOSTIC ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    historical_df, present_df = load_datasets()
    calibration, config = train_detector(historical_df)
    result_df = evaluate_detector(present_df, calibration, config)

    summary_df = generate_alert_summary(result_df)
    station_stats = generate_station_statistics(result_df)
    daily_alerts, hourly_alerts = generate_temporal_analysis(result_df)

    flag_distribution = analyze_flag_count_distribution(result_df)
    family_df, family_summary = analyze_evidence_families(result_df)
    combination_summary = analyze_evidence_combinations(family_df)
    variable_summary = analyze_variable_involvement(result_df)
    streak_df, streak_distribution = analyze_alert_streaks(result_df)

    show_alert_examples(result_df, n=10)
    plot_detector_evaluation(result_df, family_df, RESULTS_STATISTICAL_DIR)

    save_results(
        result_df=result_df,
        summary_df=summary_df,
        station_stats=station_stats,
        daily_alerts=daily_alerts,
        hourly_alerts=hourly_alerts,
        flag_distribution=flag_distribution,
        family_summary=family_summary,
        combination_summary=combination_summary,
        variable_summary=variable_summary,
        streak_df=streak_df,
        streak_distribution=streak_distribution,
        output_dir=RESULTS_STATISTICAL_DIR,
    )

    print("\n" + "=" * 70)
    print(f"Evaluation and diagnostic analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to:\n{RESULTS_STATISTICAL_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()