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
    """
    Configuration for synthetic anomaly injection.

    Key design principle:
    ---------------------
    Magnitudes are calibrated relative to LOCAL normal behaviour rather than
    fixed absolute values. This makes anomalies comparable across variables,
    stations and weather regimes.

    anomaly_rate:
        Approximate fraction of rows that should contain at least one anomaly.

    calibration_window:
        Number of historical observations used to estimate local normal
        behaviour before an injected event.

    min_context:
        Minimum observations required for local calibration. If insufficient,
        station-level fallback statistics are used.
    """

    anomaly_rate: float = 0.01
    random_seed: int = 42

    min_duration: int = 1
    max_duration: int = 24

    calibration_window: int = 72
    min_context: int = 12

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

    stuck_requires_variation: bool = True

    enforce_physical_bounds: bool = True


VARIABLE_PARAMETERS = {
    "temperature": {
        "range": (-10.0, 55.0),
        "fallback_scale": 0.5,
        "fallback_delta_scale": 0.5,
    },
    "humidity": {
        "range": (0.0, 100.0),
        "fallback_scale": 3.0,
        "fallback_delta_scale": 3.0,
    },
    "pressure": {
        "range": (950.0, 1060.0),
        "fallback_scale": 0.5,
        "fallback_delta_scale": 0.5,
    },
}

SEVERITY_STRENGTH = {
    "spike": {
        "low": (2.0, 3.0),
        "medium": (3.5, 5.0),
        "high": (5.5, 8.0),
    },
    "offset": {
        "low": (1.5, 2.5),
        "medium": (3.0, 5.0),
        "high": (5.5, 8.0),
    },
    "drift": {
        "low": (1.5, 2.5),
        "medium": (3.0, 5.0),
        "high": (5.5, 8.0),
    },
    "noise": {
        "low": (2.0, 3.0),
        "medium": (3.5, 5.5),
        "high": (6.0, 9.0),
    },
    "rate_change": {
        "low": (2.0, 3.0),
        "medium": (3.5, 5.0),
        "high": (5.5, 8.0),
    },
}

STUCK_DURATION_MULTIPLIER = {
    "low": (0.15, 0.30),
    "medium": (0.35, 0.70),
    "high": (0.75, 1.00),
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

    normalized_strength: float

    local_scale: float

    local_delta_scale: float

    total_displacement: float

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

    if len(df) == 0:
        raise ValueError("Cannot inject anomalies into an empty dataset.")


def _sample_weighted_choice(
    rng: np.random.Generator,
    weights: Dict[str, float],
) -> str:
    names = list(weights.keys())

    probabilities = np.array(
        list(weights.values()),
        dtype=float,
    )

    if np.any(probabilities < 0):
        raise ValueError("Weights cannot be negative.")

    if probabilities.sum() == 0:
        raise ValueError("At least one weight must be positive.")

    probabilities = probabilities / probabilities.sum()

    return str(rng.choice(names, p=probabilities))


def _sample_strength(
    rng: np.random.Generator,
    anomaly_type: str,
    severity: str,
) -> float:
    low, high = SEVERITY_STRENGTH[anomaly_type][severity]

    return float(rng.uniform(low, high))


def _random_sign(
    rng: np.random.Generator,
) -> int:
    return int(rng.choice([-1, 1]))


def _robust_scale(
    values: np.ndarray,
    fallback: float,
) -> float:
    """
    Robust estimate of standard deviation using MAD.

    sigma ≈ 1.4826 × MAD

    MAD is much less sensitive to existing outliers than std().
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return float(fallback)

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.std(values)

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = fallback

    return float(max(scale, fallback * 0.1, 1e-6))


def _quantile_scale(
    values: np.ndarray,
    fallback: float,
    quantile: float = 0.95,
) -> float:
    """
    Robust high-normal-change scale.

    Used for spikes, drift and rate-change calibration.

    We care about normal changes, so P95(|delta|) is often more meaningful
    than standard deviation.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float(fallback)

    scale = np.quantile(np.abs(values), quantile)

    if not np.isfinite(scale) or scale <= 1e-12:
        scale = fallback

    return float(max(scale, fallback * 0.1, 1e-6))


def _get_local_context(
    station_values: np.ndarray,
    local_start: int,
    config: AnomalyConfig,
) -> np.ndarray:
    """
    Get historical context before event start.

    Historical-only context avoids using future observations to determine
    anomaly magnitude.
    """

    start = max(0, local_start - config.calibration_window)

    context = station_values[start:local_start]

    context = context[np.isfinite(context)]

    return context


def _calculate_local_statistics(
    station_values: np.ndarray,
    local_start: int,
    variable: str,
    config: AnomalyConfig,
) -> Tuple[float, float, float]:
    """
    Returns:

        local_scale
        local_delta_scale
        local_noise_scale
    """

    fallback_scale = VARIABLE_PARAMETERS[variable]["fallback_scale"]

    fallback_delta = VARIABLE_PARAMETERS[variable]["fallback_delta_scale"]

    context = _get_local_context(
        station_values=station_values,
        local_start=local_start,
        config=config,
    )

    if len(context) < config.min_context:
        historical = station_values[:local_start]
        historical = historical[np.isfinite(historical)]

        if len(historical) >= 2:
            context = historical

    local_scale = _robust_scale(
        context,
        fallback=fallback_scale,
    )

    if len(context) >= 2:
        deltas = np.diff(context)
    else:
        deltas = np.array([])

    local_delta_scale = _quantile_scale(
        deltas,
        fallback=fallback_delta,
        quantile=0.95,
    )

    if len(deltas) >= 2:
        local_noise_scale = _robust_scale(
            deltas,
            fallback=fallback_delta,
        )
    else:
        local_noise_scale = fallback_delta

    return (
        float(local_scale),
        float(local_delta_scale),
        float(local_noise_scale),
    )


def _get_station_indices(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    return {
        str(station_id): group.index.to_numpy()
        for station_id, group in df.groupby(
            "station_id",
            sort=False,
        )
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

        local_start = int(
            rng.integers(
                0,
                max_start + 1,
            )
        )

        local_positions = range(
            local_start,
            local_start + duration,
        )

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


def _clip_to_physical_range(
    values: pd.Series,
    variable: str,
) -> pd.Series:

    lower, upper = VARIABLE_PARAMETERS[variable]["range"]

    return values.clip(
        lower=lower,
        upper=upper,
    )


def _maximum_possible_displacement(
    current_values: np.ndarray,
    variable: str,
) -> float:
    """
    Conservative physical displacement cap.

    Prevents local calibration from generating absurd values near bounds.
    """

    lower, upper = VARIABLE_PARAMETERS[variable]["range"]

    current_values = np.asarray(
        current_values,
        dtype=float,
    )

    current_values = current_values[np.isfinite(current_values)]

    if len(current_values) == 0:
        return float(upper - lower)

    upward_room = upper - np.max(current_values)
    downward_room = np.min(current_values) - lower

    return float(
        max(
            min(
                max(upward_room, 0.0),
                max(downward_room, 0.0),
            ),
            1e-6,
        )
    )


def inject_spike(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> float:
    """
    Inject one-point spike.

    Spike magnitude is calibrated relative to normal local rate-of-change.
    """

    original = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    signs = rng.choice(
        [-1, 1],
        size=len(indices),
    )

    injected = original + magnitude * signs

    df.loc[
        indices,
        variable,
    ] = injected

    df.loc[
        indices,
        variable,
    ] = _clip_to_physical_range(
        df.loc[indices, variable],
        variable,
    )

    actual = np.abs(
        df.loc[
            indices,
            variable,
        ].to_numpy(dtype=float)
        - original
    )

    return float(np.nanmax(actual))


def inject_offset(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    magnitude: float,
    rng: np.random.Generator,
) -> float:
    """
    Persistent constant bias.
    """

    sign = _random_sign(rng)

    original = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    injected = original + sign * magnitude

    df.loc[
        indices,
        variable,
    ] = injected

    df.loc[
        indices,
        variable,
    ] = _clip_to_physical_range(
        df.loc[indices, variable],
        variable,
    )

    actual = np.abs(
        df.loc[
            indices,
            variable,
        ].to_numpy(dtype=float)
        - original
    )

    return float(np.nanmax(actual))


def inject_drift(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    drift_rate: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Inject gradual accumulating sensor drift.

    drift_rate:
        Additional bias accumulated per observation.

    Returns:
        actual_total_displacement,
        actual_rate
    """

    if len(indices) == 0:
        return 0.0, 0.0

    sign = _random_sign(rng)

    original = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    steps = np.arange(len(indices))

    drift = sign * drift_rate * steps

    injected = original + drift

    df.loc[
        indices,
        variable,
    ] = injected

    df.loc[
        indices,
        variable,
    ] = _clip_to_physical_range(
        df.loc[indices, variable],
        variable,
    )

    actual_values = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    actual_displacement = actual_values - original

    total_displacement = float(np.nanmax(np.abs(actual_displacement)))

    if len(indices) > 1:
        actual_rate = float(total_displacement / (len(indices) - 1))
    else:
        actual_rate = 0.0

    return total_displacement, actual_rate


def inject_stuck(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
) -> float:
    """
    Freeze sensor at first event value.
    """

    if len(indices) == 0:
        return 0.0

    original = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    frozen_value = original[0]

    df.loc[
        indices,
        variable,
    ] = frozen_value

    displacement = np.abs(original - frozen_value)

    return float(np.nanmax(displacement))


def inject_noise(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    noise_scale: float,
    rng: np.random.Generator,
) -> float:
    """
    Add high-frequency random sensor noise.

    noise_scale is absolute standard deviation derived from local normal
    short-term variation.
    """

    original = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=len(indices),
    )

    injected = original + noise

    df.loc[
        indices,
        variable,
    ] = injected

    df.loc[
        indices,
        variable,
    ] = _clip_to_physical_range(
        df.loc[indices, variable],
        variable,
    )

    actual_values = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    actual_noise = actual_values - original

    return float(np.nanstd(actual_noise))


def inject_dropout(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
) -> float:
    """
    Simulate sensor data loss.
    """

    df.loc[
        indices,
        variable,
    ] = np.nan

    return 0.0


def inject_rate_change(
    df: pd.DataFrame,
    indices: np.ndarray,
    variable: str,
    slope_change: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Inject an abnormal change in rate-of-change.

    Unlike DRIFT, this represents a stronger dynamic change over the event.

    slope_change:
        Additional change per observation.
    """

    if len(indices) == 0:
        return 0.0, 0.0

    sign = _random_sign(rng)

    original = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    steps = np.arange(len(indices))

    ramp = sign * slope_change * steps

    injected = original + ramp

    df.loc[
        indices,
        variable,
    ] = injected

    df.loc[
        indices,
        variable,
    ] = _clip_to_physical_range(
        df.loc[indices, variable],
        variable,
    )

    actual_values = df.loc[
        indices,
        variable,
    ].to_numpy(dtype=float)

    displacement = actual_values - original

    total_displacement = float(np.nanmax(np.abs(displacement)))

    if len(indices) > 1:
        actual_rate = float(total_displacement / (len(indices) - 1))
    else:
        actual_rate = 0.0

    return total_displacement, actual_rate


def _has_meaningful_variation(
    station_values: np.ndarray,
    local_start: int,
    duration: int,
    local_scale: float,
) -> bool:
    """
    Avoid injecting a stuck anomaly into an interval that was already nearly
    constant.

    A stuck sensor is meaningful only when the clean signal would otherwise
    have varied.
    """

    end = min(
        local_start + duration,
        len(station_values),
    )

    window = station_values[local_start:end]

    window = window[np.isfinite(window)]

    if len(window) < 2:
        return False

    natural_range = np.max(window) - np.min(window)

    return bool(
        natural_range
        >= max(
            0.5 * local_scale,
            1e-6,
        )
    )


def inject_anomalies(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    config: Optional[AnomalyConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if config is None:
        config = AnomalyConfig()

    if variables is None:
        variables = [
            "temperature",
            "humidity",
            "pressure",
        ]

    _validate_input(
        df,
        variables,
    )
    rng = np.random.default_rng(config.random_seed)
    result = df.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )
    result = result.sort_values(
    ["station_id", "timestamp"]
    ).reset_index(drop=True)
    for variable in variables:
        result[variable] = pd.to_numeric(
            result[variable],
            errors="coerce",
        ).astype(float)
        result[f"{variable}_original"] = result[variable]
        
    result["synthetic_anomaly"] = False

    result["synthetic_anomaly_type"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="object",
    )

    result["synthetic_anomaly_variable"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="object",
    )

    result["synthetic_anomaly_severity"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="object",
    )

    result["synthetic_anomaly_event_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="object",
    )

    for variable in variables:

        result[f"{variable}_synthetic_anomaly"] = False

        result[f"{variable}_synthetic_anomaly_type"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="object",
        )

    target_anomalous_rows = max(
        1,
        int(len(result) * config.anomaly_rate),
    )

    station_indices = _get_station_indices(result)

    station_ids = list(station_indices.keys())

    clean_station_values: Dict[
        Tuple[str, str],
        np.ndarray,
    ] = {}

    for station_id in station_ids:

        indices = station_indices[station_id]

        for variable in variables:

            clean_station_values[(station_id, variable)] = result.loc[
                indices,
                f"{variable}_original",
            ].to_numpy(dtype=float)

    occupied: Dict[
        Tuple[str, str],
        set,
    ] = {}

    events: List[AnomalyEvent] = []

    attempts = 0

    max_attempts = max(
        target_anomalous_rows * 50,
        2000,
    )

    while (
        int(result["synthetic_anomaly"].sum()) < target_anomalous_rows
        and attempts < max_attempts
    ):

        attempts += 1

        station_id = str(rng.choice(station_ids))

        variable = str(rng.choice(variables))

        anomaly_type = _sample_weighted_choice(
            rng,
            config.anomaly_type_weights,
        )

        severity = _sample_weighted_choice(
            rng,
            config.severity_weights,
        )

        if anomaly_type == AnomalyType.SPIKE.value:

            duration = 1

        else:

            duration = int(
                rng.integers(
                    config.min_duration,
                    config.max_duration + 1,
                )
            )

        candidate_indices = station_indices[station_id]

        if config.prevent_overlap:

            local_start = _find_available_start(
                occupied=occupied,
                station_id=station_id,
                variable=variable,
                candidate_positions=candidate_indices,
                duration=duration,
                rng=rng,
            )

            if local_start is None:
                continue

        else:

            if len(candidate_indices) < duration:
                continue

            local_start = int(
                rng.integers(
                    0,
                    len(candidate_indices) - duration + 1,
                )
            )

        indices = candidate_indices[local_start : local_start + duration]

        if len(indices) == 0:
            continue

        station_values = clean_station_values[(station_id, variable)]

        (
            local_scale,
            local_delta_scale,
            local_noise_scale,
        ) = _calculate_local_statistics(
            station_values=station_values,
            local_start=local_start,
            variable=variable,
            config=config,
        )

        if anomaly_type == AnomalyType.STUCK.value and config.stuck_requires_variation:

            if not _has_meaningful_variation(
                station_values=station_values,
                local_start=local_start,
                duration=duration,
                local_scale=local_scale,
            ):
                continue

        normalized_strength = 0.0
        magnitude = 0.0

        if anomaly_type == AnomalyType.SPIKE.value:

            normalized_strength = _sample_strength(
                rng,
                "spike",
                severity,
            )

            magnitude = normalized_strength * local_delta_scale

        elif anomaly_type == AnomalyType.OFFSET.value:

            normalized_strength = _sample_strength(
                rng,
                "offset",
                severity,
            )

            magnitude = normalized_strength * local_scale

        elif anomaly_type == AnomalyType.DRIFT.value:

            normalized_strength = _sample_strength(
                rng,
                "drift",
                severity,
            )

            magnitude = normalized_strength * local_delta_scale

        elif anomaly_type == AnomalyType.NOISE.value:

            normalized_strength = _sample_strength(
                rng,
                "noise",
                severity,
            )

            magnitude = normalized_strength * local_noise_scale

        elif anomaly_type == AnomalyType.RATE_CHANGE.value:

            normalized_strength = _sample_strength(
                rng,
                "rate_change",
                severity,
            )

            magnitude = normalized_strength * local_delta_scale

        elif anomaly_type in {
            AnomalyType.STUCK.value,
            AnomalyType.DROPOUT.value,
        }:

            magnitude = 0.0
            normalized_strength = 0.0

        else:

            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        total_displacement = 0.0

        if anomaly_type == AnomalyType.SPIKE.value:

            total_displacement = inject_spike(
                result,
                indices,
                variable,
                magnitude,
                rng,
            )

        elif anomaly_type == AnomalyType.OFFSET.value:

            total_displacement = inject_offset(
                result,
                indices,
                variable,
                magnitude,
                rng,
            )

        elif anomaly_type == AnomalyType.DRIFT.value:

            (
                total_displacement,
                actual_rate,
            ) = inject_drift(
                result,
                indices,
                variable,
                magnitude,
                rng,
            )

            magnitude = actual_rate

        elif anomaly_type == AnomalyType.STUCK.value:

            total_displacement = inject_stuck(
                result,
                indices,
                variable,
            )

            magnitude = total_displacement

        elif anomaly_type == AnomalyType.NOISE.value:

            total_displacement = inject_noise(
                result,
                indices,
                variable,
                magnitude,
                rng,
            )

            magnitude = total_displacement

        elif anomaly_type == AnomalyType.DROPOUT.value:

            total_displacement = inject_dropout(
                result,
                indices,
                variable,
            )

        elif anomaly_type == AnomalyType.RATE_CHANGE.value:

            (
                total_displacement,
                actual_rate,
            ) = inject_rate_change(
                result,
                indices,
                variable,
                magnitude,
                rng,
            )

            magnitude = actual_rate

        event_id = str(uuid.uuid4())

        start_time = result.loc[
            indices[0],
            "timestamp",
        ]

        end_time = result.loc[
            indices[-1],
            "timestamp",
        ]

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
                normalized_strength=float(normalized_strength),
                local_scale=float(local_scale),
                local_delta_scale=float(local_delta_scale),
                total_displacement=float(total_displacement),
                affected_observations=len(indices),
            )
        )

        result.loc[
            indices,
            "synthetic_anomaly",
        ] = True

        result.loc[
            indices,
            f"{variable}_synthetic_anomaly",
        ] = True

        result.loc[
            indices,
            f"{variable}_synthetic_anomaly_type",
        ] = anomaly_type

        for idx in indices:

            existing_type = result.at[
                idx,
                "synthetic_anomaly_type",
            ]

            existing_variable = result.at[
                idx,
                "synthetic_anomaly_variable",
            ]

            existing_severity = result.at[
                idx,
                "synthetic_anomaly_severity",
            ]

            existing_event = result.at[
                idx,
                "synthetic_anomaly_event_id",
            ]

            if pd.isna(existing_type):

                result.at[
                    idx,
                    "synthetic_anomaly_type",
                ] = anomaly_type

                result.at[
                    idx,
                    "synthetic_anomaly_variable",
                ] = variable

                result.at[
                    idx,
                    "synthetic_anomaly_severity",
                ] = severity

                result.at[
                    idx,
                    "synthetic_anomaly_event_id",
                ] = event_id

            else:

                result.at[
                    idx,
                    "synthetic_anomaly_type",
                ] = f"{existing_type}|{anomaly_type}"

                result.at[
                    idx,
                    "synthetic_anomaly_variable",
                ] = f"{existing_variable}|{variable}"

                result.at[
                    idx,
                    "synthetic_anomaly_severity",
                ] = f"{existing_severity}|{severity}"

                result.at[
                    idx,
                    "synthetic_anomaly_event_id",
                ] = f"{existing_event}|{event_id}"

        if config.prevent_overlap:

            _mark_occupied(
                occupied=occupied,
                station_id=station_id,
                variable=variable,
                indices=indices,
            )

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
                "normalized_strength": event.normalized_strength,
                "local_scale": event.local_scale,
                "local_delta_scale": event.local_delta_scale,
                "total_displacement": event.total_displacement,
                "affected_observations": event.affected_observations,
            }
            for event in events
        ]
    )

    actual_anomalous_rows = int(result["synthetic_anomaly"].sum())

    actual_rate = actual_anomalous_rows / len(result)

    print(f"Total observations: {len(result):,}")

    print(f"Target anomalous observations: " f"{target_anomalous_rows:,}")

    print(f"Actual anomalous observations: " f"{actual_anomalous_rows:,}")

    print(f"Actual anomaly rate: " f"{actual_rate * 100:.4f}%")

    print(f"Anomaly events created: " f"{len(events_df):,}")

    print(f"Generation attempts: " f"{attempts:,}")

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

        print("\nMedian normalized strength by anomaly type:")

        strength_summary = events_df.groupby("anomaly_type")[
            "normalized_strength"
        ].median()

        print(strength_summary.to_string())

    return result, events_df


def summarize_injected_anomalies(
    contaminated_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:

    total_rows = len(contaminated_df)

    anomalous_rows = int(contaminated_df["synthetic_anomaly"].sum())

    rows = [
        {
            "metric": "total_observations",
            "value": total_rows,
        },
        {
            "metric": "anomalous_observations",
            "value": anomalous_rows,
        },
        {
            "metric": "anomaly_rate_percent",
            "value": round(
                anomalous_rows / total_rows * 100,
                4,
            ),
        },
        {
            "metric": "total_events",
            "value": len(events_df),
        },
    ]

    if not events_df.empty:

        rows.extend(
            [
                {
                    "metric": "median_event_duration_hours",
                    "value": float(events_df["duration_hours"].median()),
                },
                {
                    "metric": "median_normalized_strength",
                    "value": float(events_df["normalized_strength"].median()),
                },
            ]
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("=" * 70)
    print("LOCAL-CALIBRATED ANOMALY INJECTOR SELF TEST")
    print("=" * 70)

    rng = np.random.default_rng(0)

    n_hours = 24 * 60
    n_stations = 4

    timestamps = pd.date_range(
        "2025-01-01",
        periods=n_hours,
        freq="h",
        tz="UTC",
    )

    hours = timestamps.hour.values

    rows = []

    for i in range(n_stations):

        temperature = (
            22
            + 8 * np.sin((hours - 6) / 24 * 2 * np.pi)
            + rng.normal(
                0,
                0.5,
                n_hours,
            )
        )

        humidity = np.clip(
            60
            - 15 * np.sin((hours - 6) / 24 * 2 * np.pi)
            + rng.normal(
                0,
                3,
                n_hours,
            ),
            0,
            100,
        )

        pressure = (
            1010
            + 2 * np.sin(np.arange(n_hours) / (24 * 7) * 2 * np.pi)
            + rng.normal(0, 0.4, n_hours)
        )

        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "station_id": (f"AWS_{i:03d}"),
                    "temperature": temperature,
                    "humidity": humidity,
                    "pressure": pressure,
                }
            )
        )

    clean_df = pd.concat(
        rows,
        ignore_index=True,
    )

    variables = ["temperature", "humidity", "pressure"]

    config = AnomalyConfig(
        anomaly_rate=0.03,
        random_seed=7,
        calibration_window=72,
        min_context=12,
    )

    result, events_df = inject_anomalies(clean_df, variables=variables, config=config)

    checks = {}

    actual_rate = result["synthetic_anomaly"].mean()

    checks["Actual rate near target"] = abs(actual_rate - config.anomaly_rate) < 0.01

    stuck_ok = True

    for _, event in events_df[events_df["anomaly_type"] == "stuck"].iterrows():

        mask = (result["station_id"] == event["station_id"]) & (
            result["timestamp"].between(event["start_time"], event["end_time"])
        )

        stuck_ok &= result.loc[mask, event["variable"]].nunique() <= 1

    checks["STUCK events are constant"] = stuck_ok

    dropout_ok = True

    for _, event in events_df[events_df["anomaly_type"] == "dropout"].iterrows():

        mask = (result["station_id"] == event["station_id"]) & (
            result["timestamp"].between(event["start_time"], event["end_time"])
        )

        dropout_ok &= result.loc[mask, event["variable"]].isna().all()

    checks["DROPOUT events are NaN"] = dropout_ok

    spike_events = events_df[events_df["anomaly_type"] == "spike"]

    checks["SPIKE events have duration == 1"] = (
        (spike_events["duration_hours"] == 1).all() if len(spike_events) else True
    )

    overlap_ok = True

    for _, group in events_df.groupby(["station_id", "variable"]):

        intervals = sorted(
            zip(
                group["start_time"],
                group["end_time"],
            )
        )

        overlap_ok &= all(
            start2 > end1
            for (
                _,
                end1,
            ), (
                start2,
                _,
            ) in zip(
                intervals,
                intervals[1:],
            )
        )

    checks["No overlapping events per (station, variable)"] = overlap_ok

    range_ok = all(
        result[variable].dropna().between(*VARIABLE_PARAMETERS[variable]["range"]).all()
        for variable in variables
    )

    checks["Values respect physical range bounds"] = range_ok

    metadata_ok = True

    if not events_df.empty:

        metadata_ok &= events_df["local_scale"].notna().all()

        metadata_ok &= events_df["local_delta_scale"].notna().all()

        metadata_ok &= events_df["normalized_strength"].notna().all()

    checks["Calibration metadata recorded"] = metadata_ok

    print()

    print(
        f"Injected {len(events_df)} events, "
        f"{int(result['synthetic_anomaly'].sum())} "
        f"anomalous rows "
        f"({actual_rate * 100:.2f}% "
        f"of {len(result):,})"
    )

    print()

    all_passed = True

    for label, passed in checks.items():

        print(f"  [{'PASS' if passed else 'FAIL'}] " f"{label}")

        all_passed &= passed

    print()
    print("=" * 70)

    print(
        "OVERALL:",
        ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"),
    )

    print("=" * 70)

    if not events_df.empty:

        print("\nEVENT PREVIEW:")

        preview_columns = [
            "anomaly_type",
            "severity",
            "station_id",
            "variable",
            "duration_hours",
            "magnitude",
            "normalized_strength",
            "local_scale",
            "local_delta_scale",
            "total_displacement",
        ]

        print(events_df[preview_columns].head(10).to_string(index=False))
