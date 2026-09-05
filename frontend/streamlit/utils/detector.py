"""
Rule-based traffic-light classifier.

Swap the THRESHOLDS or evaluate_reading() body for your real detector
(z-score, ML model, whatever the SIH backend produces) without touching
the pages — they only call evaluate_station().
"""

THRESHOLDS = {
    "Temperature": {"low_red": 10, "low_yellow": 18, "high_yellow": 38, "high_red": 45, "unit": "°C"},
    "Pressure": {"low_red": 970, "low_yellow": 985, "high_yellow": 1020, "high_red": 1035, "unit": "hPa"},
    "Humidity": {"low_red": 10, "low_yellow": 20, "high_yellow": 80, "high_red": 92, "unit": "%"},
}

_ORDER = {"green": 0, "yellow": 1, "red": 2}


def evaluate_reading(sensor: str, value: float):
    t = THRESHOLDS[sensor]
    if value <= t["low_red"] or value >= t["high_red"]:
        return "red", f"{sensor} at {value:.1f}{t['unit']} is far outside the safe range ({t['low_red']}-{t['high_red']}{t['unit']})."
    if value <= t["low_yellow"] or value >= t["high_yellow"]:
        return "yellow", f"{sensor} at {value:.1f}{t['unit']} is drifting outside the normal band ({t['low_yellow']}-{t['high_yellow']}{t['unit']})."
    return "green", f"{sensor} normal at {value:.1f}{t['unit']}."


def evaluate_station(readings: dict):
    """
    readings: {"Temperature": val, "Pressure": val, "Humidity": val}
    Returns (overall_status, explanations_list, per_sensor_dict)
    overall_status is the worst of the three sensor statuses.
    """
    overall = "green"
    explanations = []
    per_sensor = {}
    for sensor, val in readings.items():
        status, msg = evaluate_reading(sensor, val)
        per_sensor[sensor] = (status, msg)
        if _ORDER[status] > _ORDER[overall]:
            overall = status
        if status != "green":
            explanations.append(msg)
    return overall, explanations, per_sensor
