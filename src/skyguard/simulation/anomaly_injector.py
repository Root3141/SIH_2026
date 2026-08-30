from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd


class AnomalyType(str, Enum):
    SPIKE = "spike"
    OFFSET = "offset"
    DRIFT = "drift"
    STUCK = "stuck"
    NOISE = "noise"
    DROPOUT = "dropout"
    RATE_CHANGE = "rate_change"


@dataclass
class AnomalyConfig:
    anomaly_rate: float = 0.01
    random_seed: int = 42
    min_duration: int = 1
    max_duration: int = 24
    anomaly_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "spike": 0.20,
            "offset": 0.20,
            "drift": 0.15,
            "stuck": 0.15,
            "noise": 0.10,
            "dropout": 0.05,
            "rate_change": 0.15,
        }
    )
    severity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.25,
        }
    )
    prevent_overlap: bool = True
    station_local_only: bool = True


VARIABLE_PARAMETERS = {
    "temperature": {
        "range": (-10.0, 55.0),
        "spike": {"low": (2.0, 4.0), "medium": (4.0, 8.0), "high": (8.0, 15.0)},
        "offset": {"low": (1.0, 2.0), "medium": (2.0, 5.0), "high": (5.0, 10.0)},
        "drift": {"low": (0.05, 0.10), "medium": (0.10, 0.30), "high": (0.30, 0.60)},
        "noise_multiplier": {
            "low": (1.5, 2.0),
            "medium": (2.0, 4.0),
            "high": (4.0, 7.0),
        },
        "rate_change": {"low": (2.0, 4.0), "medium": (4.0, 7.0), "high": (7.0, 12.0)},
    },
    "humidity": {
        "range": (0.0, 100.0),
        "spike": {"low": (8.0, 15.0), "medium": (15.0, 30.0), "high": (30.0, 50.0)},
        "offset": {"low": (5.0, 10.0), "medium": (10.0, 20.0), "high": (20.0, 35.0)},
        "drift": {"low": (0.3, 0.7), "medium": (0.7, 1.5), "high": (1.5, 3.0)},
        "noise_multiplier": {
            "low": (1.5, 2.0),
            "medium": (2.0, 4.0),
            "high": (4.0, 7.0),
        },
        "rate_change": {
            "low": (8.0, 15.0),
            "medium": (15.0, 30.0),
            "high": (30.0, 50.0),
        },
    },
    "pressure": {
        "range": (950.0, 1060.0),
        "spike": {"low": (1.0, 2.0), "medium": (2.0, 5.0), "high": (5.0, 10.0)},
        "offset": {"low": (1.0, 2.0), "medium": (2.0, 5.0), "high": (5.0, 10.0)},
        "drift": {"low": (0.03, 0.08), "medium": (0.08, 0.20), "high": (0.20, 0.50)},
        "noise_multiplier": {
            "low": (1.5, 2.0),
            "medium": (2.0, 4.0),
            "high": (4.0, 7.0),
        },
        "rate_change": {"low": (1.0, 2.0), "medium": (2.0, 4.0), "high": (4.0, 8.0)},
    },
}


@dataclass
class AnomalyEvent:
    event_id: str
    anomaly_type: str
    severity: str
    station_id: str
    variable: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: int
    magnitude: float
    affected_observations: int


def _validate_input(df: pd.DataFrame, variables: List[str]) -> None:
    required_columns = {"station_id", "timestamp", *variables}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    unsupported = set(variables) - set(VARIABLE_PARAMETERS)
    if unsupported:
        raise ValueError(
            f"Unsupported variables: {sorted(unsupported)}. "
            f"Supported variables: {sorted(VARIABLE_PARAMETERS)}"
        )


def _sample_weighted_choice(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    names = list(weights.keys())
    probabilities = np.array(list(weights.values()), dtype=float)
    if np.any(probabilities < 0):
        raise ValueError("Weights cannot be negative.")
    if probabilities.sum() == 0:
        raise ValueError("At least one weight must be positive.")
    probabilities = probabilities / probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def _sample_magnitude(
    rng: np.random.Generator, variable: str, anomaly_type: str, severity: str
) -> float:
    low, high = VARIABLE_PARAMETERS[variable][anomaly_type][severity]
    return float(rng.uniform(low, high))


def _random_sign(rng: np.random.Generator) -> int:
    return int(rng.choice([-1, 1]))


def _get_station_indices(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        str(station_id): group.index.to_numpy()
        for station_id, group in df.groupby("station_id", sort=False)
    }


def _find_available_start(
    occupied: Dict[Tuple[str, str], set],
    station_id: str,
    variable: str,
    candidate_positions: np.ndarray,
    duration: int,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> Optional[int]:
    if len(candidate_positions) < duration:
        return None
    key = (station_id, variable)
    occupied.setdefault(key, set())
    max_start = len(candidate_positions) - duration
    for _ in range(max_attempts):
        local_start = int(rng.integers(0, max_start + 1))
        local_positions = range(local_start, local_start + duration)
        global_indices = candidate_positions[list(local_positions)]
        if not any(int(idx) in occupied[key] for idx in global_indices):
            return local_start
    return None


def _mark_occupied(
    occupied: Dict[Tuple[str, str], set],
    station_id: str,
    variable: str,
    indices: np.ndarray,
) -> None:
    key = (station_id, variable)
    occupied.setdefault(key, set())
    occupied[key].update(int(idx) for idx in indices)


def _clip_to_physical_range(values: pd.Series, variable: str) -> pd.Series:
    lower, upper = VARIABLE_PARAMETERS[variable]["range"]
    return values.clip(lower=lower, upper=upper)


def inject_spike(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> None:
    signs = rng.choice([-1, 1], size=len(indices))
    df.loc[indices, variable] = df.loc[indices, variable] + magnitude * signs
    df.loc[indices, variable] = _clip_to_physical_range(
        df.loc[indices, variable], variable
    )


def inject_offset(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> None:
    sign = _random_sign(rng)
    df.loc[indices, variable] = df.loc[indices, variable] + sign * magnitude
    df.loc[indices, variable] = _clip_to_physical_range(
        df.loc[indices, variable], variable
    )


def inject_drift(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> None:
    sign = _random_sign(rng)
    drift = np.linspace(0.0, sign * magnitude * len(indices), len(indices))
    df.loc[indices, variable] = df.loc[indices, variable].to_numpy() + drift
    df.loc[indices, variable] = _clip_to_physical_range(
        df.loc[indices, variable], variable
    )


def inject_stuck(df: pd.DataFrame, indices: np.ndarray, variable: str) -> None:
    if len(indices) == 0:
        return
    frozen_value = df.loc[indices[0], variable]
    df.loc[indices, variable] = frozen_value


def inject_noise(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> None:
    values = df.loc[indices, variable]
    baseline_std = values.std()
    if pd.isna(baseline_std) or baseline_std == 0:
        baseline_std = max(abs(values.mean()) * 0.01, 0.1)
    noise_scale = baseline_std * magnitude
    noise = rng.normal(loc=0.0, scale=noise_scale, size=len(indices))
    df.loc[indices, variable] = values.to_numpy() + noise
    df.loc[indices, variable] = _clip_to_physical_range(
        df.loc[indices, variable], variable
    )


def inject_dropout(df: pd.DataFrame, indices: np.ndarray, variable: str) -> None:
    df.loc[indices, variable] = np.nan


def inject_rate_change(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> None:
    if len(indices) == 0:
        return
    sign = _random_sign(rng)
    ramp = np.linspace(0.0, sign * magnitude, len(indices))
    values = df.loc[indices, variable].to_numpy()
    df.loc[indices, variable] = values + ramp
    df.loc[indices, variable] = _clip_to_physical_range(
        df.loc[indices, variable], variable
    )


def inject_anomalies(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    config: Optional[AnomalyConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = AnomalyConfig()
    if variables is None:
        variables = ["temperature", "humidity", "pressure"]

    _validate_input(df, variables)
    rng = np.random.default_rng(config.random_seed)

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    for variable in variables:
        result[f"{variable}_original"] = result[variable]

    result["synthetic_anomaly"] = False
    result["synthetic_anomaly_type"] = pd.Series(
        pd.NA, index=result.index, dtype="object"
    )
    result["synthetic_anomaly_variable"] = pd.Series(
        pd.NA, index=result.index, dtype="object"
    )
    result["synthetic_anomaly_severity"] = pd.Series(
        pd.NA, index=result.index, dtype="object"
    )
    result["synthetic_anomaly_event_id"] = pd.Series(
        pd.NA, index=result.index, dtype="object"
    )

    for variable in variables:
        result[f"{variable}_synthetic_anomaly"] = False
        result[f"{variable}_synthetic_anomaly_type"] = pd.Series(
            pd.NA, index=result.index, dtype="object"
        )

    target_anomalous_rows = max(1, int(len(result) * config.anomaly_rate))
    station_indices = _get_station_indices(result)
    station_ids = list(station_indices.keys())
    occupied: Dict[Tuple[str, str], set] = {}
    events: List[AnomalyEvent] = []
    affected_rows = 0
    attempts = 0
    max_attempts = max(target_anomalous_rows * 20, 1000)

    while affected_rows < target_anomalous_rows and attempts < max_attempts:
        attempts += 1
        station_id = str(rng.choice(station_ids))
        variable = str(rng.choice(variables))
        anomaly_type = _sample_weighted_choice(rng, config.anomaly_type_weights)
        severity = _sample_weighted_choice(rng, config.severity_weights)

        if anomaly_type == AnomalyType.SPIKE.value:
            duration = 1
        else:
            duration = int(rng.integers(config.min_duration, config.max_duration + 1))

        candidate_indices = station_indices[station_id]

        if config.prevent_overlap:
            local_start = _find_available_start(
                occupied, station_id, variable, candidate_indices, duration, rng
            )
            if local_start is None:
                continue
        else:
            if len(candidate_indices) < duration:
                continue
            local_start = int(rng.integers(0, len(candidate_indices) - duration + 1))

        indices = candidate_indices[local_start : local_start + duration]
        if len(indices) == 0:
            continue

        if anomaly_type == AnomalyType.STUCK.value:
            magnitude = 0.0
        elif anomaly_type == AnomalyType.DROPOUT.value:
            magnitude = 0.0
        else:
            magnitude_key = (
                anomaly_type
                if anomaly_type != AnomalyType.NOISE.value
                else "noise_multiplier"
            )
            magnitude = _sample_magnitude(rng, variable, magnitude_key, severity)

        if anomaly_type == AnomalyType.SPIKE.value:
            inject_spike(result, indices, variable, magnitude, rng)
        elif anomaly_type == AnomalyType.OFFSET.value:
            inject_offset(result, indices, variable, magnitude, rng)
        elif anomaly_type == AnomalyType.DRIFT.value:
            inject_drift(result, indices, variable, magnitude, rng)
        elif anomaly_type == AnomalyType.STUCK.value:
            inject_stuck(result, indices, variable)
        elif anomaly_type == AnomalyType.NOISE.value:
            inject_noise(result, indices, variable, magnitude, rng)
        elif anomaly_type == AnomalyType.DROPOUT.value:
            inject_dropout(result, indices, variable)
        elif anomaly_type == AnomalyType.RATE_CHANGE.value:
            inject_rate_change(result, indices, variable, magnitude, rng)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        event_id = str(uuid.uuid4())
        start_time = result.loc[indices[0], "timestamp"]
        end_time = result.loc[indices[-1], "timestamp"]

        events.append(
            AnomalyEvent(
                event_id=event_id,
                anomaly_type=anomaly_type,
                severity=severity,
                station_id=station_id,
                variable=variable,
                start_time=start_time,
                end_time=end_time,
                duration_hours=len(indices),
                magnitude=float(magnitude),
                affected_observations=len(indices),
            )
        )

        result.loc[indices, "synthetic_anomaly"] = True
        result.loc[indices, f"{variable}_synthetic_anomaly"] = True
        result.loc[indices, f"{variable}_synthetic_anomaly_type"] = anomaly_type

        for idx in indices:
            existing_type = result.at[idx, "synthetic_anomaly_type"]
            existing_variable = result.at[idx, "synthetic_anomaly_variable"]
            existing_severity = result.at[idx, "synthetic_anomaly_severity"]
            existing_event = result.at[idx, "synthetic_anomaly_event_id"]

            if pd.isna(existing_type):
                result.at[idx, "synthetic_anomaly_type"] = anomaly_type
                result.at[idx, "synthetic_anomaly_variable"] = variable
                result.at[idx, "synthetic_anomaly_severity"] = severity
                result.at[idx, "synthetic_anomaly_event_id"] = event_id
            else:
                result.at[idx, "synthetic_anomaly_type"] = (
                    f"{existing_type}|{anomaly_type}"
                )
                result.at[idx, "synthetic_anomaly_variable"] = (
                    f"{existing_variable}|{variable}"
                )
                result.at[idx, "synthetic_anomaly_severity"] = (
                    f"{existing_severity}|{severity}"
                )
                result.at[idx, "synthetic_anomaly_event_id"] = (
                    f"{existing_event}|{event_id}"
                )

        if config.prevent_overlap:
            _mark_occupied(occupied, station_id, variable, indices)

        affected_rows = int(result["synthetic_anomaly"].sum())

    events_df = pd.DataFrame(
        [
            {
                "event_id": event.event_id,
                "anomaly_type": event.anomaly_type,
                "severity": event.severity,
                "station_id": event.station_id,
                "variable": event.variable,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "duration_hours": event.duration_hours,
                "magnitude": event.magnitude,
                "affected_observations": event.affected_observations,
            }
            for event in events
        ]
    )

    actual_anomalous_rows = int(result["synthetic_anomaly"].sum())
    actual_rate = actual_anomalous_rows / len(result)
    print(f"Total observations: {len(result):,}")
    print(f"Anomalous observations: {actual_anomalous_rows:,}")
    print(f"Actual anomaly rate: {actual_rate * 100:.4f}%")
    print(f"Anomaly events created: {len(events_df):,}")
    print(f"Generation attempts: {attempts:,}")

    if not events_df.empty:
        print("\nEvents by anomaly type:")
        print(events_df["anomaly_type"].value_counts().to_string())
        print("\nEvents by severity:")
        print(events_df["severity"].value_counts().to_string())
        print("\nAffected observations by variable:")
        counts = {
            variable: int(result[f"{variable}_synthetic_anomaly"].sum())
            for variable in variables
        }
        print(pd.Series(counts).to_string())

    return result, events_df


def summarize_injected_anomalies(
    contaminated_df: pd.DataFrame, events_df: pd.DataFrame
) -> pd.DataFrame:
    total_rows = len(contaminated_df)
    anomalous_rows = int(contaminated_df["synthetic_anomaly"].sum())
    return pd.DataFrame(
        [
            {"metric": "total_observations", "value": total_rows},
            {"metric": "anomalous_observations", "value": anomalous_rows},
            {
                "metric": "anomaly_rate_percent",
                "value": round(anomalous_rows / total_rows * 100, 4),
            },
            {"metric": "total_events", "value": len(events_df)},
        ]
    )


if __name__ == "__main__":
    print("=" * 70)
    print("ANOMALY INJECTOR SELF TEST")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n_hours, n_stations = 24 * 60, 4
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    hours = timestamps.hour.values

    rows = []
    for i in range(n_stations):
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "station_id": f"AWS_{i:03d}",
                    "temperature": 22
                    + 8 * np.sin((hours - 6) / 24 * 2 * np.pi)
                    + rng.normal(0, 0.5, n_hours),
                    "humidity": np.clip(55 + rng.normal(0, 3, n_hours), 0, 100),
                    "pressure": 1010 + rng.normal(0, 0.5, n_hours),
                }
            )
        )
    clean_df = pd.concat(rows, ignore_index=True)

    variables = ["temperature", "humidity", "pressure"]
    config = AnomalyConfig(anomaly_rate=0.03, random_seed=7)
    result, events_df = inject_anomalies(clean_df, variables=variables, config=config)

    checks = {}

    actual_rate = result["synthetic_anomaly"].mean()
    checks["Actual rate near target"] = abs(actual_rate - config.anomaly_rate) < 0.01

    stuck_ok = True
    for _, e in events_df[events_df["anomaly_type"] == "stuck"].iterrows():
        mask = (result["station_id"] == e["station_id"]) & result["timestamp"].between(
            e["start_time"], e["end_time"]
        )
        stuck_ok &= result.loc[mask, e["variable"]].nunique() == 1
    checks["STUCK events are constant"] = stuck_ok

    dropout_ok = True
    for _, e in events_df[events_df["anomaly_type"] == "dropout"].iterrows():
        mask = (result["station_id"] == e["station_id"]) & result["timestamp"].between(
            e["start_time"], e["end_time"]
        )
        dropout_ok &= result.loc[mask, e["variable"]].isna().all()
    checks["DROPOUT events are NaN"] = dropout_ok

    spike_events = events_df[events_df["anomaly_type"] == "spike"]
    checks["SPIKE events have duration == 1"] = (
        (spike_events["duration_hours"] == 1).all() if len(spike_events) else True
    )

    overlap_ok = True
    for _, group in events_df.groupby(["station_id", "variable"]):
        intervals = sorted(zip(group["start_time"], group["end_time"]))
        overlap_ok &= all(s2 > e1 for (_, e1), (s2, _) in zip(intervals, intervals[1:]))
    checks["No overlapping events per (station, variable)"] = overlap_ok

    range_ok = all(
        result[v].dropna().between(*VARIABLE_PARAMETERS[v]["range"]).all()
        for v in variables
    )
    checks["Values respect physical range bounds"] = range_ok

    print(
        f"\nInjected {len(events_df)} events, {int(result['synthetic_anomaly'].sum())} anomalous rows "
        f"({actual_rate*100:.2f}% of {len(result):,})\n"
    )
    all_passed = True
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_passed &= passed

    print("\n" + "=" * 70)
    print("OVERALL:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 70)
