"""Concrete fusion strategies.

Every strategy only depends on the generic Dict[str, DetectorEvidence]
contract from base.py, so none of these need to change when a detector is
added or removed - they just see more or fewer keys in the dict at
runtime.
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import DetectorEvidence, FusionResult, FusionStrategy
from .config import DETECTOR_REGISTRY


class OrStrategy(FusionStrategy):
    """Alert if ANY available detector alerts.

    High recall / precision-cost baseline (DEVELOPER_GUIDE Experiment 1).
    """

    name = "or"

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        alert = any(ev.alert for ev in evidence.values())
        score = max((ev.score for ev in evidence.values()), default=0.0)
        return FusionResult(alert=alert, score=score, strategy=self.name, contributing_detectors=evidence)


class AndStrategy(FusionStrategy):
    """Alert only if ALL available detectors agree. Conservative, low FPR."""

    name = "and"

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        if not evidence:
            return FusionResult(alert=False, score=0.0, strategy=self.name)
        alert = all(ev.alert for ev in evidence.values())
        score = min((ev.score for ev in evidence.values()), default=0.0)
        return FusionResult(alert=alert, score=score, strategy=self.name, contributing_detectors=evidence)


class KOfNStrategy(FusionStrategy):
    """Alert if at least k of the available detectors agree.

    k=2 is DEVELOPER_GUIDE Experiment 2 ("any 2 of 3"), and generalizes
    automatically if a 4th/5th detector is registered later.
    """

    def __init__(self, k: int):
        self.k = k
        self.name = f"{k}_of_n"

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        n_alerts = sum(1 for ev in evidence.values() if ev.alert)
        alert = n_alerts >= self.k
        score = n_alerts / max(len(evidence), 1)
        return FusionResult(alert=alert, score=score, strategy=self.name, contributing_detectors=evidence)


class WeightedScoreStrategy(FusionStrategy):
    """Weighted sum of normalized detector scores vs. a threshold.

    Weights default to each detector's DetectorSpec.weight, or can be
    overridden per-instance. This is DEVELOPER_GUIDE Experiment 3.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        name: str = "weighted",
    ):
        self.threshold = threshold
        self.weights = weights or {}
        self.name = name

    def _weight_for(self, detector_name: str) -> float:
        if detector_name in self.weights:
            return self.weights[detector_name]
        spec = DETECTOR_REGISTRY.get(detector_name)
        return spec.weight if spec else 1.0

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        if not evidence:
            return FusionResult(alert=False, score=0.0, strategy=self.name)
        total_weight = sum(self._weight_for(n) for n in evidence) or 1.0
        weighted_score = sum(self._weight_for(n) * ev.score for n, ev in evidence.items()) / total_weight
        return FusionResult(
            alert=weighted_score >= self.threshold,
            score=weighted_score,
            strategy=self.name,
            contributing_detectors=evidence,
        )