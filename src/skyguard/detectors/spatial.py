"""
Spatial consistency detector for weather station observations.
It compares each station to nearby stations at the same timestamp.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


@dataclass
class SpatialConfig:
    k_neighbors: int = 4
    idw_power: float = 2.0
    suspicious_percentile: float = 95.0
    anomaly_percentile: float = 99.0
    min_neighbors_required: int = 2


@dataclass
class SpatialCalibration:
    neighbors: Dict[str, List[Tuple[str, float]]]
    residual_thresholds: Dict[str, Dict[str, Dict[str, float]]]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def build_neighbor_weights(
    station_coords: Dict[str, Tuple[float, float]],
    config: SpatialConfig,
) -> Dict[str, List[Tuple[str, float]]]:
    station_ids = list(station_coords.keys())
    neighbors: Dict[str, List[Tuple[str, float]]] = {}

    for station_id in station_ids:
        lat1, lon1 = station_coords[station_id]
        distances = []
        for other_id in station_ids:
            if other_id == station_id:
                continue
            lat2, lon2 = station_coords[other_id]
            dist = haversine_km(lat1, lon1, lat2, lon2)
            distances.append((other_id, dist))

        distances.sort(key=lambda x: x[1])
        nearest = distances[: config.k_neighbors]
        weights = [1.0 / max(d, 1e-6) ** config.idw_power for _, d in nearest]
        total_weight = sum(weights)
        normalized = [
            (station_id_, w / total_weight)
            for (station_id_, _), w in zip(nearest, weights)
        ]
        neighbors[station_id] = normalized

    return neighbors


def calculate_idw_expected(
    df: pd.DataFrame,
    variable: str,
    neighbors: Dict[str, List[Tuple[str, float]]],
) -> Tuple[pd.Series, pd.Series]:
    pivot = df.pivot_table(index="timestamp", columns="station_id", values=variable)
    station_ids = pivot.columns.tolist()

    expected_matrix = pd.DataFrame(index=pivot.index, columns=station_ids, dtype=float)
    count_matrix = pd.DataFrame(index=pivot.index, columns=station_ids, dtype=float)

    for station_id in station_ids:
        neighbor_list = neighbors.get(station_id, [])
        if not neighbor_list:
            expected_matrix[station_id] = np.nan
            count_matrix[station_id] = 0
            continue

        neighbor_ids = [n for n, _ in neighbor_list]
        neighbor_weights = np.array([w for _, w in neighbor_list])
        available_ids = [n for n in neighbor_ids if n in pivot.columns]

        if not available_ids:
            expected_matrix[station_id] = np.nan
            count_matrix[station_id] = 0
            continue

        neighbor_values = pivot[available_ids]
        valid_mask = neighbor_values.notna()
        count_matrix[station_id] = valid_mask.sum(axis=1)

        weight_lookup = dict(zip(neighbor_ids, neighbor_weights))
        weights_aligned = np.array([weight_lookup[n] for n in available_ids])

        weighted_vals = neighbor_values.fillna(0) * weights_aligned
        weight_sums = (valid_mask * weights_aligned).sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            expected_matrix[station_id] = weighted_vals.sum(axis=1) / weight_sums.replace(0, np.nan)

    row_index = pd.MultiIndex.from_arrays([df["timestamp"], df["station_id"]])

    expected_long = expected_matrix.reset_index(names="timestamp").melt(
        id_vars="timestamp", var_name="station_id", value_name="expected"
    ).set_index(["timestamp", "station_id"])["expected"]
    expected = pd.Series(
        expected_long.reindex(row_index).values, index=df.index, dtype=float
    )

    count_long = count_matrix.reset_index(names="timestamp").melt(
        id_vars="timestamp", var_name="station_id", value_name="count"
    ).set_index(["timestamp", "station_id"])["count"]
    neighbor_count = pd.Series(
        count_long.reindex(row_index).fillna(0).values, index=df.index, dtype=int
    )

    return expected, neighbor_count


def calibrate_spatial_detector(
    df: pd.DataFrame,
    variables: List[str],
    station_coords: Dict[str, Tuple[float, float]],
    config: Optional[SpatialConfig] = None,
) -> SpatialCalibration:
    if config is None:
        config = SpatialConfig()

    required_columns = {"station_id", "timestamp", *variables}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    neighbors = build_neighbor_weights(station_coords, config)

    residual_thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variable in variables:
        expected, neighbor_count = calculate_idw_expected(data, variable, neighbors)
        residual = (data[variable] - expected).abs()

        for station_id, station_df in data.assign(_residual=residual, _count=neighbor_count).groupby("station_id"):
            station_valid = station_df.loc[station_df["_count"] >= config.min_neighbors_required, "_residual"]
            if len(station_valid) < 30:
                continue
            residual_thresholds.setdefault(str(station_id), {})
            residual_thresholds[str(station_id)][variable] = {
                "p95": float(station_valid.quantile(config.suspicious_percentile / 100)),
                "p99": float(station_valid.quantile(config.anomaly_percentile / 100)),
            }

    return SpatialCalibration(neighbors=neighbors, residual_thresholds=residual_thresholds)


def run_spatial_detector(
    df: pd.DataFrame,
    variables: List[str],
    calibration: SpatialCalibration,
    config: Optional[SpatialConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = SpatialConfig()

    required_columns = {"station_id", "timestamp", *variables}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    p99_cols = []
    p95_cols = []

    for variable in variables:
        expected, neighbor_count = calculate_idw_expected(result, variable, calibration.neighbors)
        result[f"{variable}_spatial_expected"] = expected
        result[f"{variable}_spatial_residual"] = result[variable] - expected
        result[f"{variable}_spatial_neighbor_count"] = neighbor_count

        p95_thresh = result["station_id"].map(
            lambda sid: calibration.residual_thresholds.get(str(sid), {}).get(variable, {}).get("p95", np.nan)
        )
        p99_thresh = result["station_id"].map(
            lambda sid: calibration.residual_thresholds.get(str(sid), {}).get(variable, {}).get("p99", np.nan)
        )

        abs_residual = result[f"{variable}_spatial_residual"].abs()
        enough_neighbors = neighbor_count >= config.min_neighbors_required

        suspicious_col = f"{variable}_spatial_suspicious"
        anomaly_col = f"{variable}_spatial_anomaly"
        result[suspicious_col] = enough_neighbors & (abs_residual > p95_thresh)
        result[anomaly_col] = enough_neighbors & (abs_residual > p99_thresh)

        p95_cols.append(suspicious_col)
        p99_cols.append(anomaly_col)

    result["spatial_family_count"] = result[p95_cols].fillna(False).astype(bool).sum(axis=1)
    p99_count = result[p99_cols].fillna(False).astype(bool).sum(axis=1)

    result["spatial_severity_level"] = np.select(
        condlist=[
            p99_count >= 1,
            result["spatial_family_count"] >= 1,
        ],
        choicelist=["anomaly", "suspicious"],
        default="normal",
    )
    result["spatial_alert"] = result["spatial_severity_level"] == "anomaly"

    return result


def _run_self_test():
    print("=" * 70)
    print("SPATIAL DETECTOR SELF TEST")
    print("=" * 70)

    rng = np.random.default_rng(0)
    station_coords = {
        "AWS_A": (28.61, 77.21),
        "AWS_B": (28.46, 77.03),
        "AWS_C": (28.54, 77.39),
        "AWS_D": (28.41, 77.32),
    }

    n_hours = 24 * 60
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    regional_temp = 20 + 8 * np.sin(np.arange(n_hours) / 24 * 2 * np.pi)

    rows = []
    for station_id in station_coords:
        local_noise = rng.normal(0, 0.8, n_hours)
        temp = regional_temp + local_noise
        rows.append(pd.DataFrame({"timestamp": timestamps, "station_id": station_id, "temperature": temp}))

    df = pd.concat(rows, ignore_index=True)

    config = SpatialConfig(k_neighbors=3)
    calibration = calibrate_spatial_detector(df, ["temperature"], station_coords, config)

    test_df = df.copy()

    spike_time = timestamps[500]
    isolated_mask = (test_df["station_id"] == "AWS_A") & (test_df["timestamp"] == spike_time)
    test_df.loc[isolated_mask, "temperature"] += 15

    regional_time = timestamps[800]
    regional_mask = test_df["timestamp"] == regional_time
    test_df.loc[regional_mask, "temperature"] += 6

    result = run_spatial_detector(test_df, ["temperature"], calibration, config)

    isolated_row = result[(result["station_id"] == "AWS_A") & (result["timestamp"] == spike_time)]
    regional_rows = result[result["timestamp"] == regional_time]

    print("\nCase A: isolated single-station spike (AWS_A, +15C, neighbors normal)")
    print(isolated_row[["temperature", "temperature_spatial_expected", "temperature_spatial_residual", "spatial_severity_level", "spatial_alert"]].to_string(index=False))
    a_pass = bool(isolated_row["spatial_alert"].iloc[0])
    print(f"  Expected: flagged (True)  ->  {'PASS' if a_pass else 'FAIL'}")

    print("\nCase B: regional event (all 4 stations +6C together, same hour)")
    print(regional_rows[["station_id", "temperature", "temperature_spatial_expected", "temperature_spatial_residual", "spatial_severity_level", "spatial_alert"]].to_string(index=False))
    b_pass = not regional_rows["spatial_alert"].any()
    print(f"  Expected: NOT flagged (all False)  ->  {'PASS' if b_pass else 'FAIL'}")

    print("\n" + "=" * 70)
    print("OVERALL:", "ALL CHECKS PASSED" if (a_pass and b_pass) else "SOME CHECKS FAILED")
    print("=" * 70)

    return result


if __name__ == "__main__":
    _run_self_test()