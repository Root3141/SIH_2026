"""Detector registry: the single place that lists which detectors exist
and how to read their output columns.

To add a new detector to fusion once it's ready, add ONE DetectorSpec here
(or call register_detector() at runtime). No other fusion file needs to
change - adapters, strategies, and the engine all just iterate over
whatever detectors are registered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class DetectorSpec:
    name: str
    alert_col: str
    score_col: Optional[str] = None
    severity_col: Optional[str] = None
    # Optional custom normalizer if a detector's raw score isn't already
    # 0-1 (e.g. LSTM reconstruction error needs min-max/percentile scaling
    # relative to its own calibration). Identity by default.
    normalize: Callable[[float], float] = field(default=lambda x: x)
    weight: float = 1.0  # used by WeightedScoreStrategy

    _severity_map: Dict[str, float] = field(
        default_factory=lambda: {
            "normal": 0.0,
            "suspicious": 0.33,
            "anomaly": 0.67,
            "critical": 1.0,
        }
    )

    def severity_to_float(self, value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return self._severity_map.get(str(value).lower(), 0.5)


# --- Current detector registry -------------------------------------------
# Column names taken from your DEVELOPER_GUIDE.md / CODEBASE_REFERENCE.md
# conventions (detector_name_alert, detector_name_score, etc).
#
# lstm_ae_score is a raw reconstruction error, NOT already 0-1 - the
# `normalize` lambda below is a placeholder clip. Swap it for something
# that scales relative to the calibration P95/P99 thresholds stored in
# calibration.pkl once you wire that in (e.g. score / P99, clipped to 1.0).

DETECTOR_REGISTRY: Dict[str, DetectorSpec] = {
    "statistical": DetectorSpec(
        name="statistical",
        alert_col="statistical_alert",
        severity_col="statistical_severity_label",
    ),
    "spatial": DetectorSpec(
        name="spatial",
        alert_col="spatial_alert",
        severity_col="spatial_severity_level",
    ),
    "lstm_ae": DetectorSpec(
        name="lstm_ae",
        alert_col="lstm_ae_alert_original",  # confirmed from evaluate_combined.py output
        score_col=None,  # TODO: unverified - set the real reconstruction-error column
        # name once known (evaluate_combined.py compares it against
        # calibration P95/P99: 0.008555 / 0.017892). Until then this
        # detector contributes a binary 0/1 signal like the other two,
        # which is fine for the count-based tier strategy below but
        # limits the weighted/logistic strategies to alert-only info.
        severity_col=None,
        normalize=lambda x: min(max(x, 0.0), 1.0),
    ),
}


def register_detector(spec: DetectorSpec) -> None:
    """Add a new detector to the registry at runtime.

    Example (adding a hypothetical 4th detector, no other fusion file
    needs to be touched):

        from skyguard.fusion.config import register_detector, DetectorSpec
        register_detector(DetectorSpec(
            name="humidity_physics",
            alert_col="humidity_physics_alert",
            score_col="humidity_physics_score",
        ))
    """
    DETECTOR_REGISTRY[spec.name] = spec