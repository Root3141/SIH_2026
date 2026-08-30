"""
Statistical anomaly detection for weather station observations.

This module provides lightweight, explainable anomaly detection using:
1. Operational range checks
2. Rate-of-change checks
3. Persistence / frozen sensor detection
4. Rolling Z-score detection

Calibration stage learns ROC thresholds and hour-of-day baselines from historical data.
Detection stage applies frozen calibration and generates anomaly evidence.
Station-level statistical evidence only (spatial and ML detection handled separately).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StatisticalConfig:
    """Configuration for statistical anomaly detection."""

    zscore_windows: List[int] = field(default_factory=lambda: [24, 168])
    min_periods: int = 6
    zscore_threshold: float = 3.5
    roc_percentile: float = 99.0
    persistence_window: int = 6
    persistence_tolerance: Dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 0.05,
            "humidity": 0.05,
            "pressure": 0.05,
        }
    )


RANGE_BOUNDS = {
    "temperature": (-10.0, 55.0),
    "humidity": (0.0, 100.0),
    "pressure": (950.0, 1060.0),
}


@dataclass
class StatisticalCalibration:
    """
    Learned statistical baselines from historical normal data.

    Attributes:
        roc_thresholds: station_id -> variable -> maximum normal rate of change
        hourly_baselines: station_id -> hour -> variable -> expected value
    """

    roc_thresholds: Dict[str, Dict[str, float]]
    hourly_baselines: Dict[str, Dict[int, Dict[str, float]]]


def calibrate_statistical_detector(
    df: pd.DataFrame, variables: List[str], config: Optional[StatisticalConfig] = None
) -> StatisticalCalibration:
    """
    Calibrate statistical detector using historical normal data.

    Should only receive clean training data (2023-01-01 to 2025-12-31).
    """
    if config is None:
        config = StatisticalConfig()

    required_columns = {"station_id", "timestamp", *variables}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    roc_thresholds: Dict[str, Dict[str, float]] = {}
    for station_id, station_df in data.groupby("station_id"):
        station_thresholds = {}
        for variable in variables:
            delta = station_df[variable].diff().abs()
            threshold = delta.quantile(config.roc_percentile / 100)
            station_thresholds[variable] = float(threshold)
        roc_thresholds[str(station_id)] = station_thresholds

    data["_hour"] = data["timestamp"].dt.hour
    hourly_baselines: Dict[str, Dict[int, Dict[str, float]]] = {}
    grouped = data.groupby(["station_id", "_hour"], sort=False)

    for (station_id, hour), group in grouped:
        station_id = str(station_id)
        hour = int(hour)
        if station_id not in hourly_baselines:
            hourly_baselines[station_id] = {}
        hourly_baselines[station_id][hour] = {
            variable: float(group[variable].mean()) for variable in variables
        }

    return StatisticalCalibration(
        roc_thresholds=roc_thresholds, hourly_baselines=hourly_baselines
    )


def detect_range_anomalies(series: pd.Series, variable: str) -> pd.Series:
    """Detect values outside configured operational sanity bounds."""
    if variable not in RANGE_BOUNDS:
        raise ValueError(f"No range bounds configured for '{variable}'")
    lower, upper = RANGE_BOUNDS[variable]
    return (series < lower) | (series > upper)


def calculate_rate_of_change(df: pd.DataFrame, variable: str) -> pd.Series:
    """Calculate absolute rate of change between consecutive observations per station."""
    return df.groupby("station_id")[variable].diff().abs()


def detect_roc_anomalies(
    df: pd.DataFrame, variable: str, calibration: StatisticalCalibration
) -> pd.Series:
    """Detect rate-of-change anomalies using frozen calibration thresholds."""
    delta = calculate_rate_of_change(df, variable)
    thresholds = df["station_id"].map(
        lambda station_id: calibration.roc_thresholds.get(str(station_id), {}).get(
            variable, np.nan
        )
    )
    return delta > thresholds


def calculate_hourly_residual(
    df: pd.DataFrame, variable: str, calibration: StatisticalCalibration
) -> pd.Series:
    """Calculate residual from calibrated hour-of-day baseline (observed - expected)."""
    hours = df["timestamp"].dt.hour
    expected_values = pd.Series(
        [
            calibration.hourly_baselines.get(str(station_id), {})
            .get(int(hour), {})
            .get(variable, np.nan)
            for station_id, hour in zip(df["station_id"], hours)
        ],
        index=df.index,
        dtype=float,
    )
    return df[variable] - expected_values


def calculate_rolling_zscore(
    df: pd.DataFrame, variable: str, residual: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """
    Calculate causal rolling Z-score.

    Rolling mean/std are shifted by one timestep so observation at time t
    is compared only against observations from times < t.
    """
    grouped = residual.groupby(df["station_id"], group_keys=False)

    rolling_mean = grouped.transform(
        lambda series: (
            series.rolling(window=window, min_periods=min(min_periods, window))
            .mean()
            .shift(1)
        )
    )

    rolling_std = grouped.transform(
        lambda series: (
            series.rolling(window=window, min_periods=min(min_periods, window))
            .std()
            .shift(1)
        )
    )

    rolling_std = rolling_std.replace(0, np.nan)
    zscore = (residual - rolling_mean) / rolling_std
    return zscore.abs()


def detect_persistence(
    df: pd.DataFrame, variable: str, config: StatisticalConfig
) -> pd.Series:
    """Detect frozen/stuck sensors by checking rolling std per station."""
    tolerance = config.persistence_tolerance.get(variable, 0.05)
    rolling_std = df.groupby("station_id")[variable].transform(
        lambda series: series.rolling(
            window=config.persistence_window,
            min_periods=config.persistence_window,
        ).std()
    )
    return rolling_std < tolerance


def run_statistical_detector(
    df: pd.DataFrame,
    variables: List[str],
    calibration: StatisticalCalibration,
    config: Optional[StatisticalConfig] = None,
) -> pd.DataFrame:
    """
    Run statistical anomaly detection.

    Returns original dataframe enriched with range/ROC/residual/Z-score/persistence
    anomaly flags, plus statistical_flag_count, statistical_severity, and statistical_alert.
    """
    if config is None:
        config = StatisticalConfig()

    required_columns = {"station_id", "timestamp", *variables}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    evidence_columns = []

    for variable in variables:
        range_col = f"{variable}_range_anomaly"
        result[range_col] = detect_range_anomalies(result[variable], variable)
        evidence_columns.append(range_col)

        roc_value_col = f"{variable}_rate_of_change"
        result[roc_value_col] = calculate_rate_of_change(result, variable)

        roc_flag_col = f"{variable}_roc_anomaly"
        result[roc_flag_col] = detect_roc_anomalies(result, variable, calibration)
        evidence_columns.append(roc_flag_col)

        residual_col = f"{variable}_hourly_residual"
        result[residual_col] = calculate_hourly_residual(result, variable, calibration)

        for window in config.zscore_windows:
            zscore_col = f"{variable}_zscore_{window}h"
            result[zscore_col] = calculate_rolling_zscore(
                result,
                variable,
                result[residual_col],
                window=window,
                min_periods=config.min_periods,
            )
            zscore_flag_col = f"{variable}_zscore_{window}h_anomaly"
            result[zscore_flag_col] = result[zscore_col] > config.zscore_threshold
            evidence_columns.append(zscore_flag_col)

        persistence_col = f"{variable}_persistence_anomaly"
        result[persistence_col] = detect_persistence(result, variable, config)
        evidence_columns.append(persistence_col)

    result["statistical_flag_count"] = (
        result[evidence_columns].fillna(False).astype(bool).sum(axis=1)
    )
    result["statistical_severity"] = result["statistical_flag_count"] / len(
        evidence_columns
    )
    result["statistical_alert"] = result["statistical_flag_count"] > 0

    return result


def summarize_statistical_alerts(
    df: pd.DataFrame,
    variables: List[str],
) -> pd.DataFrame:
    """Produce a summary of statistical detector activity."""
    rows = []
    for variable in variables:
        for column in df.columns:
            if column.startswith(f"{variable}_") and column.endswith("_anomaly"):
                count = df[column].fillna(False).astype(bool).sum()
                rows.append(
                    {
                        "variable": variable,
                        "detector": column,
                        "alert_count": int(count),
                        "alert_rate": (count / len(df)),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pass
