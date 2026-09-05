"""
LSTM Autoencoder Detector Evaluation
====================================

File: src/skyguard/evaluation/evaluate_lstm_autoencoder.py

This evaluator supports three modes:

    full     Train/calibrate, inject, score, evaluate, and cache artifacts.
    score    Load cached calibration, inject fresh anomalies, score, evaluate,
             and cache the scored evaluation results. No retraining.
    analyze  Load cached scored results and run diagnostics/threshold sweeps.
             No retraining and no LSTM inference.

Typical workflow:

    First run after changing the model:
        python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode full

    Re-run evaluation data with the same trained model:
        python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode score

    Cheap threshold/diagnostic experiments:
        python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode analyze

The cached artifacts live under results/lstm_autoencoder/:
    calibration.pkl          fitted model + normalization/calibration state
    scored_evaluation.parquet  one inference result that can be re-analyzed cheaply

Important:
    Threshold sweeps and clean-vs-anomalous diagnostics use the synthetic
    evaluation labels to measure behavior. They do not change model training.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skyguard.detectors.lstm_autoencoder import (
    LSTMAEConfig,
    calibrate_lstm_autoencoder_detector,
    run_lstm_autoencoder_detector,
)

try:
    from skyguard.config.paths import HISTORICAL_PARQUET, PRESENT_PARQUET, RESULTS_DIR
except ImportError:  # pragma: no cover - fallback for standalone use
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    HISTORICAL_PARQUET = (
        _PROJECT_ROOT / "data" / "raw" / "ncr_weather_historical.parquet"
    )
    PRESENT_PARQUET = (
        _PROJECT_ROOT / "data" / "raw" / "ncr_weather_2026_present.parquet"
    )
    RESULTS_DIR = _PROJECT_ROOT / "results"

try:
    from skyguard.simulation.anomaly_injector import AnomalyConfig, inject_anomalies

    _INJECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _INJECTOR_AVAILABLE = False

OUTPUT_DIR = Path(RESULTS_DIR) / "lstm_autoencoder"
CALIBRATION_CACHE = OUTPUT_DIR / "calibration.pkl"
SCORED_CACHE = OUTPUT_DIR / "scored_evaluation.parquet"
VARIABLES = ["temperature", "pressure", "humidity"]
ANOMALY_RATE = 0.02
DIAGNOSTIC_SCORE_PERCENTILES = [90.0, 95.0, 97.0, 98.0, 99.0, 99.5, 99.9]
OVERLAP_BUFFER_MULTIPLIER = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SkyGuard LSTM autoencoder."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "score", "analyze"],
        default="full",
        help=(
            "full=train+score+analyze, score=reuse calibration and rerun inference, "
            "analyze=reuse cached scores only"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the relevant cache and recompute that stage.",
    )
    return parser.parse_args()


def save_calibration(calibration) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_CACHE, "wb") as f:
        pickle.dump(calibration, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved calibration cache: {CALIBRATION_CACHE}\n")


def load_calibration():
    if not CALIBRATION_CACHE.exists():
        raise FileNotFoundError(
            f"Calibration cache not found: {CALIBRATION_CACHE}\n"
            "Run --mode full first."
        )
    with open(CALIBRATION_CACHE, "rb") as f:
        calibration = pickle.load(f)
    print(f"Loaded calibration cache: {CALIBRATION_CACHE}")
    print(f"  Backend:              {calibration.backend}")
    print(f"  Stations calibrated:  {calibration.n_stations}")
    print(f"  Training windows:     {calibration.n_training_windows}")
    print(f"  Validation windows:   {calibration.n_validation_windows}")
    print(f"  Calibration P95:      {calibration.threshold_p95:.5f}")
    print(f"  Calibration P99:      {calibration.threshold_p99:.5f}\n")
    return calibration, calibration.config


def save_scored_results(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(SCORED_CACHE)
    print(f"Saved scored evaluation cache: {SCORED_CACHE}\n")


def load_scored_results() -> pd.DataFrame:
    if not SCORED_CACHE.exists():
        raise FileNotFoundError(
            f"Scored evaluation cache not found: {SCORED_CACHE}\n"
            "Run --mode full or --mode score first."
        )
    results = pd.read_parquet(SCORED_CACHE)
    results["timestamp"] = pd.to_datetime(results["timestamp"], utc=True)
    print(f"Loaded scored evaluation cache: {SCORED_CACHE}")
    print(f"  Rows: {len(results):,}")
    print(f"  Scored rows: {int(results['lstm_ae_scored'].sum()):,}\n")
    return results


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("LOADING DATASETS")
    historical = pd.read_parquet(HISTORICAL_PARQUET)
    present = pd.read_parquet(PRESENT_PARQUET)

    for df in (historical, present):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values(["station_id", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"Historical calibration data:\n  Rows:     {len(historical):,}")
    print(f"  Stations: {historical['station_id'].nunique()}")
    print(
        f"  Period:   {historical['timestamp'].min()} to {historical['timestamp'].max()}"
    )
    print(f"\nUnseen evaluation data:\n  Rows:     {len(present):,}")
    print(f"  Stations: {present['station_id'].nunique()}")
    print(f"  Period:   {present['timestamp'].min()} to {present['timestamp'].max()}\n")
    return historical, present


def train_or_load_calibration(historical: pd.DataFrame, force: bool = False):
    if CALIBRATION_CACHE.exists() and not force:
        return load_calibration()

    print("CALIBRATING LSTM AUTOENCODER DETECTOR")
    config = LSTMAEConfig(variables=VARIABLES)
    calibration = calibrate_lstm_autoencoder_detector(historical, config)
    print(f"  Backend used:        {calibration.backend}")
    print(f"  Stations calibrated: {calibration.n_stations}")
    print(f"  Training windows:    {calibration.n_training_windows}")
    print(f"  Validation windows:  {calibration.n_validation_windows}")
    print(f"  Threshold P95:       {calibration.threshold_p95:.5f}")
    print(f"  Threshold P99:       {calibration.threshold_p99:.5f}")
    print("Calibration complete\n")
    save_calibration(calibration)
    return calibration, config


def inject_evaluation_anomalies(
    present: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("INJECTING SYNTHETIC ANOMALIES")
    if not _INJECTOR_AVAILABLE:
        raise ImportError(
            "skyguard.simulation.anomaly_injector is not importable in this environment."
        )

    injector_config = AnomalyConfig(anomaly_rate=ANOMALY_RATE)
    corrupted, injection_log = inject_anomalies(present, VARIABLES, injector_config)

    corrupted = corrupted.rename(
        columns={
            "synthetic_anomaly": "is_anomaly",
            "synthetic_anomaly_event_id": "anomaly_id",
            "synthetic_anomaly_type": "anomaly_type",
            "synthetic_anomaly_variable": "anomaly_variable",
            "synthetic_anomaly_severity": "anomaly_severity",
        }
    )

    required = {
        "is_anomaly",
        "anomaly_id",
        "anomaly_type",
        "anomaly_variable",
        "anomaly_severity",
    }
    missing = required - set(corrupted.columns)
    if missing:
        raise ValueError(
            f"Anomaly injection did not produce expected columns: {missing}"
        )

    total = len(corrupted)
    anomalous = int(corrupted["is_anomaly"].sum())
    n_events = (
        injection_log["event_id"].nunique()
        if "event_id" in injection_log
        else len(injection_log)
    )
    print(f"Total observations:      {total:,}")
    print(f"Anomalous observations:  {anomalous:,}")
    print(f"Actual anomaly rate:     {anomalous / total:.2%}")
    print(f"Anomaly events created:  {n_events}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_dir = Path(RESULTS_DIR).parent / "data" / "synthetic"
    save_path = (
        synthetic_dir / "lstm_ae_evaluation_injected.parquet"
        if synthetic_dir.exists()
        else OUTPUT_DIR / "injected.parquet"
    )
    corrupted.to_parquet(save_path)
    return corrupted, injection_log


def run_detector(contaminated: pd.DataFrame, calibration, config) -> pd.DataFrame:
    print("RUNNING LSTM AUTOENCODER DETECTOR")
    results = run_lstm_autoencoder_detector(contaminated, calibration, config)
    total = len(results)
    scored = int(results["lstm_ae_scored"].sum())
    alerts = int(results["lstm_ae_alert"].sum())
    print(f"Total observations:        {total:,}")
    print(f"Scored observations:       {scored:,} ({scored / total:.1%} coverage)")
    print(f"Current built-in alerts:   {alerts:,}")
    print(f"Current alert rate:        {alerts / max(scored, 1):.3%}\n")
    save_scored_results(results)
    return results


def calculate_metrics_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    subset: pd.DataFrame | None = None,
) -> dict:
    scored = (
        subset
        if subset is not None
        else df[df["lstm_ae_scored"] & df["lstm_ae_score"].notna()]
    )

    if len(scored) == 0:
        return {
            "threshold": float(threshold),
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "false_positive_rate": 0.0,
            "alert_count": 0,
            "alert_rate": 0.0,
            "rows_scored": 0,
        }

    y_true = scored["is_anomaly"].astype(bool).to_numpy()
    scores = scored["lstm_ae_score"].to_numpy(dtype=float)
    y_pred = scores > threshold

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    alert_count = int(y_pred.sum())

    return {
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "alert_count": alert_count,
        "alert_rate": alert_count / len(scored),
        "rows_scored": len(scored),
    }


def calculate_binary_metrics(df: pd.DataFrame) -> dict:
    scored = df[df["lstm_ae_scored"]]
    y_true = scored["is_anomaly"].astype(bool)
    y_pred = scored["lstm_ae_alert"].astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "rows_scored": len(scored),
        "rows_unscored": len(df) - len(scored),
    }


def evaluate_observation_level(df: pd.DataFrame) -> pd.DataFrame:
    print("OBSERVATION-LEVEL PERFORMANCE (CURRENT BUILT-IN THRESHOLD)")
    metrics = calculate_binary_metrics(df)
    print(f"True Positives:  {metrics['true_positives']:,}")
    print(f"False Positives: {metrics['false_positives']:,}")
    print(f"False Negatives: {metrics['false_negatives']:,}")
    print(f"True Negatives:  {metrics['true_negatives']:,}")
    print(f"Unscored rows:   {metrics['rows_unscored']:,}")
    print(f"\nPrecision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1 Score:            {metrics['f1_score']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}\n")
    return pd.DataFrame([metrics])


def build_threshold_candidates(df: pd.DataFrame, calibration) -> pd.DataFrame:
    scored = df[df["lstm_ae_scored"] & df["lstm_ae_score"].notna()]
    scores = scored["lstm_ae_score"].to_numpy(dtype=float)
    clean_scores = scored.loc[~scored["is_anomaly"], "lstm_ae_score"].to_numpy(
        dtype=float
    )

    candidates: list[tuple[str, float]] = [
        ("calibration_p95", float(calibration.threshold_p95)),
        ("calibration_p99", float(calibration.threshold_p99)),
    ]

    for p in DIAGNOSTIC_SCORE_PERCENTILES:
        candidates.append((f"eval_all_p{p:g}", float(np.percentile(scores, p))))

    for p in DIAGNOSTIC_SCORE_PERCENTILES:
        candidates.append((f"eval_clean_p{p:g}", float(np.percentile(clean_scores, p))))

    return (
        pd.DataFrame(candidates, columns=["threshold_label", "threshold"])
        .drop_duplicates(subset=["threshold"])
        .sort_values("threshold")
        .reset_index(drop=True)
    )


def evaluate_events_at_threshold(
    df: pd.DataFrame, threshold: float
) -> tuple[int, int, float]:
    anomalies = df[
        df["is_anomaly"] & df["lstm_ae_scored"] & df["lstm_ae_score"].notna()
    ]
    if anomalies.empty:
        return 0, 0, 0.0

    detected_events = 0
    total_events = 0
    for _, group in anomalies.groupby("anomaly_id"):
        total_events += 1
        if bool((group["lstm_ae_score"] > threshold).any()):
            detected_events += 1

    recall = detected_events / total_events if total_events else 0.0
    return detected_events, total_events, recall


def run_threshold_sweep(
    df: pd.DataFrame,
    calibration,
    include_buffer_clear: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    print("THRESHOLD SWEEP")
    candidates = build_threshold_candidates(df, calibration)

    all_records = []
    clear_records = []

    for row in candidates.itertuples(index=False):
        metrics = calculate_metrics_at_threshold(df, row.threshold)
        detected_events, total_events, event_recall = evaluate_events_at_threshold(
            df, row.threshold
        )
        metrics.update(
            {
                "threshold_label": row.threshold_label,
                "detected_events": detected_events,
                "total_events": total_events,
                "event_recall": event_recall,
            }
        )
        all_records.append(metrics)

        if include_buffer_clear and "rows_from_nearest_anomaly" in df.columns:
            clear = df[
                df["lstm_ae_scored"]
                & df["lstm_ae_score"].notna()
                & (~df["is_anomaly"] | (df["rows_from_nearest_anomaly"] > 0))
            ].copy()
            clear_records.append(
                calculate_metrics_at_threshold(df, row.threshold, subset=clear)
                | {
                    "threshold_label": row.threshold_label,
                    "detected_events": np.nan,
                    "total_events": np.nan,
                    "event_recall": np.nan,
                }
            )

    sweep = pd.DataFrame(all_records)
    clear_sweep = pd.DataFrame(clear_records) if clear_records else None

    display_cols = [
        "threshold_label",
        "threshold",
        "alert_rate",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "event_recall",
    ]
    print(
        sweep[display_cols].to_string(
            index=False,
            formatters={
                "threshold": "{:.6f}".format,
                "alert_rate": "{:.3%}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
                "false_positive_rate": "{:.4%}".format,
                "event_recall": "{:.4f}".format,
            },
        )
    )
    print()
    return sweep, clear_sweep


def summarize_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["lstm_ae_scored"] & df["lstm_ae_score"].notna()].copy()
    groups = {
        "all_scored": scored["lstm_ae_score"].to_numpy(dtype=float),
        "clean": scored.loc[~scored["is_anomaly"], "lstm_ae_score"].to_numpy(
            dtype=float
        ),
        "anomalous": scored.loc[scored["is_anomaly"], "lstm_ae_score"].to_numpy(
            dtype=float
        ),
    }

    rows = []
    percentiles = [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]
    for group_name, values in groups.items():
        values = values[np.isfinite(values)]
        row = {
            "group": group_name,
            "count": len(values),
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "std": float(np.std(values)) if len(values) else np.nan,
            "min": float(np.min(values)) if len(values) else np.nan,
            "max": float(np.max(values)) if len(values) else np.nan,
        }
        for p in percentiles:
            row[f"p{p:g}"] = float(np.percentile(values, p)) if len(values) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    print("SCORE DISTRIBUTION")
    print(summary.to_string(index=False))
    print()
    return summary


def score_distribution_by_station(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["lstm_ae_scored"] & df["lstm_ae_score"].notna()].copy()
    clean = scored[~scored["is_anomaly"]]
    summary = (
        clean.groupby("station_id")["lstm_ae_score"]
        .agg(
            clean_observations="size",
            mean_score="mean",
            median_score="median",
            p95_score=lambda s: np.percentile(s, 95),
            p99_score=lambda s: np.percentile(s, 99),
            max_score="max",
        )
        .reset_index()
    )
    return summary.sort_values("p99_score", ascending=False)


def add_distance_to_nearest_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row distance to nearest injected anomaly, separately per station."""
    out = df.copy()
    out["rows_from_nearest_anomaly"] = np.inf

    for station_id, group in out.groupby("station_id", sort=False):
        group = group.sort_values("timestamp")
        anomaly_mask = group["is_anomaly"].astype(bool).to_numpy()
        distances = np.full(len(group), np.inf)

        last_anomaly = None
        for i in range(len(group)):
            if anomaly_mask[i]:
                last_anomaly = i
            if last_anomaly is not None:
                distances[i] = i - last_anomaly

        next_anomaly = None
        for i in range(len(group) - 1, -1, -1):
            if anomaly_mask[i]:
                next_anomaly = i
            if next_anomaly is not None:
                distances[i] = min(distances[i], next_anomaly - i)

        out.loc[group.index, "rows_from_nearest_anomaly"] = distances

    return out


def analyze_window_overlap_contamination(
    df: pd.DataFrame,
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("WINDOW-OVERLAP CONTAMINATION DIAGNOSTIC")

    scored = df[df["lstm_ae_scored"] & df["lstm_ae_score"].notna()].copy()
    buffer = window_size - 1

    anomalous = scored[scored["is_anomaly"]]
    near_anomaly_clean = scored[
        (~scored["is_anomaly"]) & (scored["rows_from_nearest_anomaly"] <= buffer)
    ]
    buffer_clear_clean = scored[
        (~scored["is_anomaly"]) & (scored["rows_from_nearest_anomaly"] > buffer)
    ]

    groups = {
        "anomalous": anomalous,
        "near_anomaly_clean": near_anomaly_clean,
        "buffer_clear_clean": buffer_clear_clean,
    }

    rows = []
    for name, subset in groups.items():
        scores = subset["lstm_ae_score"].dropna().to_numpy(dtype=float)
        if len(scores) == 0:
            continue
        rows.append(
            {
                "group": name,
                "count": len(scores),
                "median": float(np.percentile(scores, 50)),
                "p90": float(np.percentile(scores, 90)),
                "p95": float(np.percentile(scores, 95)),
                "p99": float(np.percentile(scores, 99)),
                "p99_5": float(np.percentile(scores, 99.5)),
                "mean": float(np.mean(scores)),
                "max": float(np.max(scores)),
            }
        )

    summary = pd.DataFrame(rows)
    print(f"Window size: {window_size}")
    print(f"Overlap buffer: ±{buffer} rows")
    print()
    print(summary.to_string(index=False))
    print()

    for label, subset in [
        ("near_anomaly_clean", near_anomaly_clean),
        ("buffer_clear_clean", buffer_clear_clean),
    ]:
        if len(subset) == 0:
            continue
        current = (subset["lstm_ae_score"] > 0).mean()
        print(f"{label}: {len(subset):,} rows")
        print(f"  Mean score: {subset['lstm_ae_score'].mean():.6f}")
        print(
            f"  Alert rate at calibration P95 ({0.00855:.5f}): "
            f"{(subset['lstm_ae_score'] > 0.00855).mean():.2%}"
        )
        _ = current

    return summary, near_anomaly_clean, buffer_clear_clean


def temporal_score_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = df[
        df["lstm_ae_scored"]
        & df["lstm_ae_score"].notna()
        & (~df["is_anomaly"])
        & (df["rows_from_nearest_anomaly"] > 23)
    ].copy()
    if clean.empty:
        return pd.DataFrame(), pd.DataFrame()

    clean["month"] = clean["timestamp"].dt.month
    clean["hour"] = clean["timestamp"].dt.hour

    monthly = (
        clean.groupby("month")["lstm_ae_score"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
        )
        .reset_index()
    )
    hourly = (
        clean.groupby("hour")["lstm_ae_score"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
        )
        .reset_index()
    )

    print("TEMPORAL SCORE DIAGNOSTICS (BUFFER-CLEAR CLEAN)")
    print("By month:")
    print(monthly.to_string(index=False))
    print("\nBy hour of day:")
    print(hourly.to_string(index=False))
    print()
    return monthly, hourly


def per_variable_diagnostics(df: pd.DataFrame, calibration) -> pd.DataFrame:
    clean = df[
        df["lstm_ae_scored"]
        & df["lstm_ae_score"].notna()
        & (~df["is_anomaly"])
        & (df["rows_from_nearest_anomaly"] > 23)
    ]
    rows = []

    for var in VARIABLES:
        error_col = f"{var}_lstm_ae_error"
        if error_col not in df.columns:
            continue
        values = clean[error_col].dropna()
        thresholds = calibration.per_variable_thresholds.get(var, {})
        rows.append(
            {
                "variable": var,
                "count": len(values),
                "eval_clean_mean": float(values.mean()),
                "eval_clean_p95": float(values.quantile(0.95)),
                "eval_clean_p99": float(values.quantile(0.99)),
                "calibration_p95": float(thresholds.get("p95", np.nan)),
                "calibration_p99": float(thresholds.get("p99", np.nan)),
            }
        )

    summary = pd.DataFrame(rows)
    print("PER-VARIABLE RECONSTRUCTION ERROR (BUFFER-CLEAR CLEAN)")
    print(summary.to_string(index=False))
    print()
    return summary


def per_station_diagnostics(df: pd.DataFrame, calibration) -> pd.DataFrame:
    """Compare buffer-clear clean score distributions across stations."""

    clean = df[
        df["lstm_ae_scored"]
        & df["lstm_ae_score"].notna()
        & (~df["is_anomaly"])
        & (df["rows_from_nearest_anomaly"] > calibration.config.window_size - 1)
    ].copy()

    if clean.empty:
        return pd.DataFrame()

    station_scores = (
        clean.groupby("station_id")["lstm_ae_score"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
        )
        .sort_values("p99", ascending=False)
        .reset_index()
    )

    station_scores["p95_vs_calibration"] = (
        station_scores["p95"] / calibration.threshold_p95
    )

    station_scores["p99_vs_calibration"] = (
        station_scores["p99"] / calibration.threshold_p99
    )

    print("PER-STATION SCORE DIAGNOSTICS (BUFFER-CLEAR CLEAN)")
    print(station_scores.to_string(index=False))
    print()

    return station_scores


def evaluate_event_level(df: pd.DataFrame) -> pd.DataFrame:
    print("EVENT-LEVEL PERFORMANCE (CURRENT BUILT-IN THRESHOLD)")
    anomalies = df[df["is_anomaly"] & df["lstm_ae_scored"]]
    if anomalies.empty:
        print("No scored anomalous rows to evaluate at event level.\n")
        return pd.DataFrame()

    records = []
    for event_id, group in anomalies.groupby("anomaly_id"):
        detected = bool(group["lstm_ae_alert"].any())
        records.append(
            {
                "anomaly_id": event_id,
                "anomaly_type": group["anomaly_type"].iloc[0],
                "anomaly_variable": group["anomaly_variable"].iloc[0],
                "anomaly_severity": group["anomaly_severity"].iloc[0],
                "duration": len(group),
                "detected": detected,
                "observations_detected": int(group["lstm_ae_alert"].sum()),
            }
        )

    events_df = pd.DataFrame(records)
    total_events = len(events_df)
    detected_events = int(events_df["detected"].sum())
    recall = detected_events / total_events if total_events else 0.0

    print(f"Total anomaly events:    {total_events}")
    print(f"Detected anomaly events: {detected_events}")
    print(f"Missed anomaly events:   {total_events - detected_events}")
    print(f"Event-level recall:      {recall:.4f}\n")
    return events_df


def evaluate_by_anomaly_type(events_df: pd.DataFrame) -> pd.DataFrame:
    print("PERFORMANCE BY ANOMALY TYPE")
    if events_df.empty:
        return pd.DataFrame()
    summary = (
        events_df.groupby("anomaly_type")
        .agg(total_events=("detected", "size"), detected=("detected", "sum"))
        .reset_index()
    )
    summary["missed"] = summary["total_events"] - summary["detected"]
    summary["recall"] = summary["detected"] / summary["total_events"]
    summary["recall_percent"] = (summary["recall"] * 100).round(2)
    print(summary.to_string(index=False))
    print()
    return summary


def evaluate_by_variable(events_df: pd.DataFrame) -> pd.DataFrame:
    print("PERFORMANCE BY VARIABLE")
    if events_df.empty:
        return pd.DataFrame()
    summary = (
        events_df.groupby("anomaly_variable")
        .agg(total_events=("detected", "size"), detected=("detected", "sum"))
        .reset_index()
    )
    summary["missed"] = summary["total_events"] - summary["detected"]
    summary["recall"] = summary["detected"] / summary["total_events"]
    summary["recall_percent"] = (summary["recall"] * 100).round(2)
    print(summary.to_string(index=False))
    print()
    return summary


def analyze_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["lstm_ae_scored"]]
    clean = scored[~scored["is_anomaly"]]
    fp = clean[clean["lstm_ae_alert"]]

    print("FALSE POSITIVE ANALYSIS")
    print(f"Clean observations: {len(clean):,}")
    print(f"False positives:    {len(fp):,}")
    print(f"False positive rate: {len(fp) / len(clean) if len(clean) else 0.0:.4%}\n")

    by_station = (
        clean.groupby("station_id")["lstm_ae_alert"]
        .agg(clean_observations="size", false_positives="sum")
        .reset_index()
    )
    by_station["false_positive_rate"] = (
        by_station["false_positives"] / by_station["clean_observations"]
    )
    return by_station.sort_values("false_positive_rate", ascending=False)


def analyze_severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["lstm_ae_scored"]]
    summary = (
        scored.groupby("lstm_ae_severity_label")
        .agg(
            observations=("is_anomaly", "size"),
            injected_anomalies=("is_anomaly", "sum"),
        )
        .reset_index()
    )
    print("SEVERITY DISTRIBUTION (CURRENT BUILT-IN LABELS)")
    print(summary.to_string(index=False))
    print()
    return summary


def analyze_alert_policy(df: pd.DataFrame) -> None:
    """Show sensitive raw alerts separately from confirmed anomaly alerts."""

    scored = df[df["lstm_ae_scored"]].copy()

    raw_count = int(scored["lstm_ae_suspicious"].sum())
    confirmed_count = int(scored["lstm_ae_alert"].sum())

    raw_rate = scored["lstm_ae_suspicious"].mean()
    confirmed_rate = scored["lstm_ae_alert"].mean()

    print("ALERT POLICY SUMMARY")
    print(f"Raw P95+ alerts:       {raw_count:,} " f"({raw_rate:.2%})")
    print(f"Confirmed P99+ alerts: {confirmed_count:,} " f"({confirmed_rate:.2%})")
    print()


def generate_threshold_plot(df: pd.DataFrame, sweep: pd.DataFrame, calibration) -> Path:
    scored = df[df["lstm_ae_scored"] & df["lstm_ae_score"].notna()].copy()
    clean_scores = scored.loc[~scored["is_anomaly"], "lstm_ae_score"].to_numpy(
        dtype=float
    )
    anomaly_scores = scored.loc[scored["is_anomaly"], "lstm_ae_score"].to_numpy(
        dtype=float
    )

    all_scores = np.concatenate([clean_scores, anomaly_scores])
    positive_scores = all_scores[np.isfinite(all_scores) & (all_scores > 0)]
    bins = (
        np.geomspace(
            max(np.min(positive_scores) * 0.9, 1e-12),
            np.max(positive_scores) * 1.05,
            80,
        )
        if len(positive_scores)
        else 50
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.hist(
        clean_scores[clean_scores > 0],
        bins=bins,
        alpha=0.60,
        density=True,
        label="Clean",
    )
    ax.hist(
        anomaly_scores[anomaly_scores > 0],
        bins=bins,
        alpha=0.60,
        density=True,
        label="Injected anomaly",
    )
    ax.axvline(calibration.threshold_p95, linestyle="--", label="Calibration P95")
    ax.axvline(calibration.threshold_p99, linestyle="--", label="Calibration P99")
    ax.set_xscale("log")
    ax.set_xlabel("LSTM AE reconstruction score")
    ax.set_ylabel("Density")
    ax.set_title("Clean vs Anomalous Score Distribution")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(sweep["threshold"], sweep["recall"], marker="o", label="Recall")
    ax.plot(
        sweep["threshold"],
        sweep["false_positive_rate"],
        marker="o",
        label="False positive rate",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold Trade-off: Recall vs FPR")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision")
    ax.plot(sweep["threshold"], sweep["f1_score"], marker="o", label="F1")
    ax.set_xscale("log")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Trade-off: Precision / F1")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(sweep["threshold"], sweep["alert_rate"] * 100, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Alert rate (%)")
    ax.set_title("Alert Volume vs Threshold")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "lstm_ae_threshold_analysis.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def generate_current_plots(
    df: pd.DataFrame, metrics: dict, by_type: pd.DataFrame
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1], labels=["Actual Normal", "Actual Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
    ax.set_title("Current Built-in Threshold: Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    if not by_type.empty:
        ax.barh(by_type["anomaly_type"], by_type["recall_percent"])
        ax.set_xlabel("Recall (%)")
        ax.set_title("Recall by Anomaly Type")
    else:
        ax.axis("off")

    ax = axes[1, 0]
    scored = df[df["lstm_ae_scored"]]
    by_station = (
        scored.groupby("station_id")["lstm_ae_alert"].mean().sort_values() * 100
    )
    ax.barh(by_station.index.astype(str), by_station.values)
    ax.set_xlabel("Alert Rate (%)")
    ax.set_title("Current Alert Rate by Station")

    ax = axes[1, 1]
    severity_counts = scored["lstm_ae_severity_label"].value_counts()
    ax.bar(severity_counts.index, severity_counts.values)
    ax.set_title("Current Severity Distribution")
    ax.set_ylabel("Observation Count")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "lstm_ae_evaluation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_analysis_outputs(
    results: pd.DataFrame,
    obs_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    by_type: pd.DataFrame,
    by_variable: pd.DataFrame,
    fp_by_station: pd.DataFrame,
    severity_dist: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    score_distribution: pd.DataFrame,
    score_by_station: pd.DataFrame,
    overlap_diagnostics: pd.DataFrame,
    per_variable: pd.DataFrame,
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_metrics.to_csv(OUTPUT_DIR / "observation_metrics.csv", index=False)
    event_metrics.to_csv(OUTPUT_DIR / "event_metrics.csv", index=False)
    by_type.to_csv(OUTPUT_DIR / "performance_by_anomaly_type.csv", index=False)
    by_variable.to_csv(OUTPUT_DIR / "performance_by_variable.csv", index=False)
    fp_by_station.to_csv(OUTPUT_DIR / "false_positive_analysis.csv", index=False)
    severity_dist.to_csv(OUTPUT_DIR / "severity_distribution.csv", index=False)
    threshold_sweep.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
    score_distribution.to_csv(OUTPUT_DIR / "score_distribution.csv", index=False)
    score_by_station.to_csv(OUTPUT_DIR / "clean_score_by_station.csv", index=False)
    overlap_diagnostics.to_csv(
        OUTPUT_DIR / "window_overlap_diagnostics.csv", index=False
    )
    per_variable.to_csv(OUTPUT_DIR / "per_variable_error_diagnostics.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_score_diagnostics.csv", index=False)
    hourly.to_csv(OUTPUT_DIR / "hourly_score_diagnostics.csv", index=False)

    prediction_cols = [
        c
        for c in results.columns
        if c
        in (
            "station_id",
            "timestamp",
            "is_anomaly",
            "anomaly_type",
            "anomaly_variable",
        )
        or c.startswith("lstm_ae_")
        or c.endswith("_lstm_ae_expected")
        or c.endswith("_lstm_ae_residual")
    ]
    alerts_or_truth = results[results["is_anomaly"] | results["lstm_ae_alert"]]
    alerts_or_truth[prediction_cols].to_csv(
        OUTPUT_DIR / "anomaly_predictions.csv", index=False
    )


def run_analysis(results: pd.DataFrame, calibration) -> None:
    if "rows_from_nearest_anomaly" not in results.columns:
        results = add_distance_to_nearest_anomaly(results)

    obs_metrics = evaluate_observation_level(results)
    event_metrics = evaluate_event_level(results)
    by_type = evaluate_by_anomaly_type(event_metrics)
    by_variable = evaluate_by_variable(event_metrics)
    fp_by_station = analyze_false_positives(results)
    severity_dist = analyze_severity_distribution(results)
    analyze_alert_policy(results)

    score_distribution = summarize_score_distribution(results)
    score_by_station = score_distribution_by_station(results)
    overlap_diagnostics, near_clean, buffer_clear_clean = (
        analyze_window_overlap_contamination(results, calibration.config.window_size)
    )

    print("BUFFER-AWARE THRESHOLD SWEEP")
    candidates = build_threshold_candidates(results, calibration)
    records = []
    clear_records = []

    for row in candidates.itertuples(index=False):
        metrics = calculate_metrics_at_threshold(results, row.threshold)
        detected_events, total_events, event_recall = evaluate_events_at_threshold(
            results, row.threshold
        )
        metrics.update(
            {
                "threshold_label": row.threshold_label,
                "detected_events": detected_events,
                "total_events": total_events,
                "event_recall": event_recall,
            }
        )
        records.append(metrics)

        clear_metrics = calculate_metrics_at_threshold(
            results,
            row.threshold,
            subset=buffer_clear_clean,
        )
        clear_metrics["threshold_label"] = row.threshold_label
        clear_records.append(clear_metrics)

    threshold_sweep = pd.DataFrame(records)
    buffer_clear_sweep = pd.DataFrame(clear_records)

    display_cols = [
        "threshold_label",
        "threshold",
        "alert_rate",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "event_recall",
    ]
    print("\nAll scored rows:")
    print(threshold_sweep[display_cols].to_string(index=False))
    print("\nBuffer-clear clean rows only:")
    print(
        buffer_clear_sweep[
            ["threshold_label", "threshold", "alert_rate", "false_positive_rate"]
        ].to_string(index=False)
    )
    print()

    per_variable = per_variable_diagnostics(results, calibration)
    per_station = per_station_diagnostics(results, calibration)
    monthly, hourly = temporal_score_diagnostics(results)

    results.to_parquet(SCORED_CACHE)

    metrics_dict = obs_metrics.iloc[0].to_dict()
    current_plot = generate_current_plots(results, metrics_dict, by_type)
    threshold_plot = generate_threshold_plot(results, threshold_sweep, calibration)

    save_analysis_outputs(
        results=results,
        obs_metrics=obs_metrics,
        event_metrics=event_metrics,
        by_type=by_type,
        by_variable=by_variable,
        fp_by_station=fp_by_station,
        severity_dist=severity_dist,
        threshold_sweep=threshold_sweep,
        score_distribution=score_distribution,
        score_by_station=score_by_station,
        overlap_diagnostics=overlap_diagnostics,
        per_variable=per_variable,
        monthly=monthly,
        hourly=hourly,
    )
    print(f"PER STATION SCORE DIAGNOSTICS (BUFFER-CLEAR CLEAN):")
    print(per_station.to_string(index=False))
    print(f"Saved current evaluation plot:  {current_plot}")
    print(f"Saved threshold analysis plot: {threshold_plot}")
    print(f"Saved buffer-aware sweep: {OUTPUT_DIR / 'threshold_sweep.csv'}")
    print(f"Saved scored cache:          {SCORED_CACHE}\n")


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)

    print("=" * 70)
    print("LSTM AUTOENCODER DETECTOR EVALUATION")
    print(f"Mode: {args.mode}")
    print(f"Started: {started.isoformat()}")
    print("=" * 70 + "\n")

    if args.mode == "analyze":
        calibration, _config = load_calibration()
        results = load_scored_results()
        run_analysis(results, calibration)

    elif args.mode == "score":
        historical, present = load_datasets()
        calibration, config = load_calibration()
        contaminated, _injection_log = inject_evaluation_anomalies(present)
        results = run_detector(contaminated, calibration, config)
        run_analysis(results, calibration)

    else:  # full
        historical, present = load_datasets()
        calibration, config = train_or_load_calibration(historical, force=args.force)
        contaminated, _injection_log = inject_evaluation_anomalies(present)
        results = run_detector(contaminated, calibration, config)
        run_analysis(results, calibration)

    finished = datetime.now(timezone.utc)
    print("=" * 70)
    print(f"LSTM Autoencoder evaluation complete: {finished.isoformat()}")
    print(f"Duration: {finished - started}")
    print("=" * 70)


if __name__ == "__main__":
    main()
