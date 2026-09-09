"""Concrete fusion strategies, all producing a 3-class label:
NORMAL / SUSPICIOUS / ANOMALY.

Every strategy only depends on the generic Dict[str, DetectorEvidence]
contract from base.py, so none of these need to change when a detector is
added or removed.

Default thresholds below are chosen from your own
evaluate_combined.py run (2026-09-07), not guessed:

    method                  fpr      event_recall
    statistical            0.70%     85.16%
    spatial                1.89%     67.19%
    stat_spatial_OR        2.40%     91.41%
    all_three_OR_1x        7.22%     96.88%
    any_two_of_three_1x    0.82%     89.06%   <- best FPR/recall tradeoff
    all_three_AND_1x       0.17%     57.81%

any_two_of_three_1x dominates every OR-based combo on FPR while staying
within 2-8 points of their event recall, and dominates AND on recall
while barely costing any FPR. That's the empirical basis for
CountTierStrategy's defaults: 1 detector -> SUSPICIOUS (visibility
without cost), 2+ detectors -> ANOMALY (the confirmed-alert tier).
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import DetectorEvidence, FusionResult, FusionStrategy, NORMAL, SUSPICIOUS, ANOMALY
from .config import DETECTOR_REGISTRY


class CountTierStrategy(FusionStrategy):
    """Bucket by how many available detectors alert.

    n_suspicious detectors alerting -> SUSPICIOUS
    n_anomaly detectors alerting    -> ANOMALY
    Defaults (1, 2) reproduce your any_two_of_three_1x result exactly
    when run on 3 detectors: FPR 0.82%, event recall 89.06%.
    """

    def __init__(self, n_suspicious: int = 1, n_anomaly: int = 2, name: str = "count_tier"):
        self.n_suspicious = n_suspicious
        self.n_anomaly = n_anomaly
        self.name = name

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        n_alerts = sum(1 for ev in evidence.values() if ev.alert)
        score = n_alerts / max(len(evidence), 1)

        if n_alerts >= self.n_anomaly:
            label = ANOMALY
        elif n_alerts >= self.n_suspicious:
            label = SUSPICIOUS
        else:
            label = NORMAL

        return FusionResult(label=label, score=score, strategy=self.name, contributing_detectors=evidence)


class WeightedTierStrategy(FusionStrategy):
    """Bucket by a weighted sum of detector scores against two thresholds.

    Needs real per-detector scores (not just 0/1 alerts) to be more
    informative than CountTierStrategy - useful once lstm_ae's raw
    reconstruction-error column is wired into config.py's score_col.
    Weights default to each detector's DetectorSpec.weight.
    """

    def __init__(
        self,
        suspicious_threshold: float = 0.33,
        anomaly_threshold: float = 0.66,
        weights: Optional[Dict[str, float]] = None,
        name: str = "weighted_tier",
    ):
        self.suspicious_threshold = suspicious_threshold
        self.anomaly_threshold = anomaly_threshold
        self.weights = weights or {}
        self.name = name

    def _weight_for(self, detector_name: str) -> float:
        if detector_name in self.weights:
            return self.weights[detector_name]
        spec = DETECTOR_REGISTRY.get(detector_name)
        return spec.weight if spec else 1.0

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        if not evidence:
            return FusionResult(label=NORMAL, score=0.0, strategy=self.name)

        total_weight = sum(self._weight_for(n) for n in evidence) or 1.0
        weighted_score = sum(self._weight_for(n) * ev.score for n, ev in evidence.items()) / total_weight

        if weighted_score >= self.anomaly_threshold:
            label = ANOMALY
        elif weighted_score >= self.suspicious_threshold:
            label = SUSPICIOUS
        else:
            label = NORMAL

        return FusionResult(label=label, score=weighted_score, strategy=self.name, contributing_detectors=evidence)