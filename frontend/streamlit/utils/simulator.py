"""
Generates a stand-in "2026 dataset" for each station so the dashboard can be
demoed end-to-end before the real sensor backend is wired up.

Replace `generate_all_datasets()` with a loader that reads your real 2026
CSV/DB export, keeping the same shape: one DataFrame per station with
columns [timestamp, Temperature, Pressure, Humidity].
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---- 12 monitored AWS stations with lat/lon for the map view ----
STATIONS = {
    "Delhi Central": (28.6139, 77.2090),
    "Gurugram": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
    "Faridabad": (28.4089, 77.3178),
    "Ghaziabad": (28.6692, 77.4538),
    "Sonipat": (28.9931, 77.0151),
    "Rohtak": (28.8955, 76.6066),
    "Meerut": (28.9845, 77.7064),
    "Jhajjar": (28.6067, 76.6565),
    "Greater Noida": (28.4744, 77.5040),
    "Bahadurgarh": (28.6924, 76.9356),
    "Hapur": (28.7306, 77.7759),
}

# station_id -> station name, kept for reference / backend matching
STATION_IDS = {
    "AWS_001": "Delhi Central",
    "AWS_002": "Gurugram",
    "AWS_003": "Noida",
    "AWS_004": "Faridabad",
    "AWS_005": "Ghaziabad",
    "AWS_006": "Sonipat",
    "AWS_007": "Rohtak",
    "AWS_008": "Meerut",
    "AWS_009": "Jhajjar",
    "AWS_010": "Greater Noida",
    "AWS_011": "Bahadurgarh",
    "AWS_012": "Hapur",
}

BASELINES = {
    "Temperature": {"mean": 28, "std": 3},
    "Pressure": {"mean": 1005, "std": 4},
    "Humidity": {"mean": 55, "std": 10},
}

N_POINTS = 500  # length of the simulated stream per station


def _random_walk(mean, std, n, seed, spike_prob=0.03, spike_scale=4.0):
    """Mean-reverting random walk with occasional injected spikes, so the
    demo produces a realistic mix of green/yellow/red readings over time."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    values[0] = mean
    for i in range(1, n):
        step = rng.normal(0, std * 0.15)
        values[i] = values[i - 1] + step
        values[i] += (mean - values[i]) * 0.05  # pull back toward baseline
        if rng.random() < spike_prob:
            direction = rng.choice([-1, 1])
            values[i] += direction * std * spike_scale * rng.uniform(0.6, 1.4)
    return values


def generate_station_dataset(station_name: str, n: int = N_POINTS) -> pd.DataFrame:
    seed = abs(hash(station_name)) % (2**32)
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(n)]
    data = {"timestamp": timestamps}
    for i, (sensor, cfg) in enumerate(BASELINES.items()):
        data[sensor] = _random_walk(cfg["mean"], cfg["std"], n, seed + i * 17)
    return pd.DataFrame(data)


def generate_all_datasets() -> dict:
    return {name: generate_station_dataset(name) for name in STATIONS}
