"""Statistical anomaly detection for weather station observations."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StatisticalConfig:
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
    roc_thresholds: Dict[str, Dict[int, Dict[str, float]]]
    hourly_baselines: Dict[str, Dict[int, Dict[str, float]]]


def calibrate_statistical_detector(
    df: pd.DataFrame,
    variables: List[str],
    config: Optional[StatisticalConfig] = None,
) -> StatisticalCalibration:
    if config is None:
        config = StatisticalConfig()

    required = {"station_id", "timestamp", *variables}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    data["_hour"] = data["timestamp"].dt.hour

    roc_thresholds: Dict[str, Dict[int, Dict[str, float]]] = {}
    for station_id, station_df in data.groupby("station_id"):
        station_id = str(station_id)
        station_df = station_df.sort_values("timestamp")
        roc_thresholds[station_id] = {}
        for variable in variables:
            delta = station_df[variable].diff().abs()
            temp = pd.DataFrame({"hour": station_df["_hour"], "delta": delta})
            for hour, hour_df in temp.groupby("hour"):
                threshold = hour_df["delta"].quantile(config.roc_percentile / 100)
                roc_thresholds[station_id].setdefault(int(hour), {})
                roc_thresholds[station_id][int(hour)][variable] = float(threshold)

    hourly_baselines: Dict[str, Dict[int, Dict[str, float]]] = {}
    for (station_id, hour), group in data.groupby(["station_id", "_hour"], sort=False):
        station_id = str(station_id)
        hourly_baselines.setdefault(station_id, {})
        hourly_baselines[station_id][int(hour)] = {
            variable: float(group[variable].mean()) for variable in variables
        }

    return StatisticalCalibration(
        roc_thresholds=roc_thresholds,
        hourly_baselines=hourly_baselines,
    )


def detect_range_anomalies(series: pd.Series, variable: str) -> pd.Series:
    lower, upper = RANGE_BOUNDS[variable]
    return (series < lower) | (series > upper)


def calculate_rate_of_change(df: pd.DataFrame, variable: str) -> pd.Series:
    return df.groupby("station_id")[variable].diff().abs()


def detect_roc_anomalies(
    df: pd.DataFrame,
    variable: str,
    calibration: StatisticalCalibration,
) -> pd.Series:
    delta = calculate_rate_of_change(df, variable)
    hours = df["timestamp"].dt.hour
    thresholds = pd.Series(
        [
            calibration.roc_thresholds.get(str(station_id), {})
            .get(int(hour), {})
            .get(variable, np.nan)
            for station_id, hour in zip(df["station_id"], hours)
        ],
        index=df.index,
        dtype=float,
    )
    return delta > thresholds


def calculate_hourly_residual(
    df: pd.DataFrame,
    variable: str,
    calibration: StatisticalCalibration,
) -> pd.Series:
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
    df: pd.DataFrame,
    variable: str,
    residual: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    grouped = residual.groupby(df["station_id"], group_keys=False)

    rolling_mean = grouped.transform(
        lambda s: s.rolling(window=window, min_periods=min(min_periods, window))
        .mean()
        .shift(1)
    )
    rolling_std = grouped.transform(
        lambda s: s.rolling(window=window, min_periods=min(min_periods, window))
        .std()
        .shift(1)
    )
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (residual - rolling_mean) / rolling_std
    return zscore.abs()


def detect_persistence(
    df: pd.DataFrame,
    variable: str,
    config: StatisticalConfig,
) -> pd.Series:
    tolerance = config.persistence_tolerance.get(variable, 0.05)
    rolling_std = df.groupby("station_id")[variable].transform(
        lambda s: s.rolling(
            window=config.persistence_window, min_periods=config.persistence_window
        ).std()
    )
    return rolling_std < tolerance


def build_evidence_families(
    result: pd.DataFrame,
    variables: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    range_columns = [
        f"{variable}_range_anomaly"
        for variable in variables
        if f"{variable}_range_anomaly" in result.columns
    ]
    result["evidence_range"] = (
        result[range_columns].fillna(False).astype(bool).any(axis=1)
        if range_columns
        else False
    )

    roc_columns = [
        f"{variable}_roc_anomaly"
        for variable in variables
        if f"{variable}_roc_anomaly" in result.columns
    ]
    result["evidence_roc"] = (
        result[roc_columns].fillna(False).astype(bool).any(axis=1)
        if roc_columns
        else False
    )

    level_columns = []
    for variable in variables:
        level_columns.extend(
            [
                col
                for col in result.columns
                if col.startswith(f"{variable}_zscore_") and col.endswith("_anomaly")
            ]
        )
    result["evidence_level"] = (
        result[level_columns].fillna(False).astype(bool).any(axis=1)
        if level_columns
        else False
    )

    persistence_columns = [
        f"{variable}_persistence_anomaly"
        for variable in variables
        if f"{variable}_persistence_anomaly" in result.columns
    ]
    result["evidence_persistence"] = (
        result[persistence_columns].fillna(False).astype(bool).any(axis=1)
        if persistence_columns
        else False
    )

    return result, [
        "evidence_range",
        "evidence_roc",
        "evidence_level",
        "evidence_persistence",
    ]


def assign_statistical_severity(
    result: pd.DataFrame, family_columns: List[str]
) -> pd.DataFrame:
    family_matrix = result[family_columns].fillna(False).astype(bool)
    result["statistical_family_count"] = family_matrix.sum(axis=1).astype(int)
    result["statistical_raw_alert"] = result["statistical_flag_count"] > 0

    family_count = result["statistical_family_count"]
    range_flag = result["evidence_range"]
    persistence_flag = result["evidence_persistence"]

    severity = np.full(len(result), "normal", dtype=object)
    severity[family_count == 1] = "suspicious"
    severity[family_count >= 2] = "anomaly"
    severity[persistence_flag] = "anomaly"
    severity[range_flag] = "critical"

    result["statistical_severity_label"] = severity
    severity_scores = {
        "normal": 0.0,
        "suspicious": 0.33,
        "anomaly": 0.67,
        "critical": 1.0,
    }
    result["statistical_severity"] = (
        result["statistical_severity_label"].map(severity_scores).astype(float)
    )
    result["statistical_alert"] = result["statistical_severity_label"].isin(
        ["anomaly", "critical"]
    )
    return result


def run_statistical_detector(
    df: pd.DataFrame,
    variables: List[str],
    calibration: StatisticalCalibration,
    config: Optional[StatisticalConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = StatisticalConfig()

    required = {"station_id", "timestamp", *variables}
    missing = required - set(df.columns)
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
                window,
                config.min_periods,
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

    result, family_columns = build_evidence_families(result, variables)
    result = assign_statistical_severity(result, family_columns)
    return result


def summarize_statistical_alerts(
    df: pd.DataFrame,
    variables: List[str],
) -> pd.DataFrame:
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
                        "alert_rate": count / len(df) if len(df) > 0 else 0.0,
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values("alert_count", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("Statistical anomaly detector module loaded successfully.")
    print("\nDetector families: range, rate of change, level, persistence")
    print("Fusion logic: 0 families = normal, 1 = suspicious, 2+ = anomaly")
    print("Range anomalies are treated as critical.")
    print("Persistence anomalies are treated as high-confidence anomalies.")
