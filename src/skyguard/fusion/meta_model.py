"""Learned fusion: logistic regression over normalized detector scores,
bucketed into NORMAL / SUSPICIOUS / ANOMALY via two probability
thresholds.

This is still the strategy to hand to your XAI teammate. The model
itself is trained as an ordinary binary classifier (is_anomaly vs not,
same as your other evaluations), because that's what your labeled
benchmark supports. The 3-class label is then read off the predicted
probability with two thresholds - same idea as WeightedTierStrategy, but
with the score coming from a fitted model instead of a hand-picked
weighted average.

Feature names == detector names in a fixed order, so SHAP
(shap.LinearExplainer(meta.model, X)) or raw model.coef_ directly answers
"how much did each detector push this row's probability up."
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .base import DetectorEvidence, FusionResult, FusionStrategy, NORMAL, SUSPICIOUS, ANOMALY


class LogisticMetaStrategy(FusionStrategy):
    name = "logistic_meta"

    def __init__(
        self,
        detector_order: Optional[List[str]] = None,
        suspicious_threshold: float = 0.3,
        anomaly_threshold: float = 0.6,
    ):
        # detector_order fixes feature order for training/inference/SHAP.
        # If None, it's inferred from the training data on first fit().
        self.detector_order = detector_order
        self.suspicious_threshold = suspicious_threshold
        self.anomaly_threshold = anomaly_threshold
        self.model = None

    def _vectorize(self, evidence: Dict[str, DetectorEvidence]) -> np.ndarray:
        order = self.detector_order or sorted(evidence.keys())
        return np.array([evidence[d].score if d in evidence else 0.0 for d in order])

    def fit(self, evidence_rows: List[Dict[str, DetectorEvidence]], y_true: List[bool]) -> None:
        """Train on rows of evidence dicts + ground-truth is_anomaly labels
        from the same injected benchmark evaluate_combined.py uses, so this
        is directly comparable to the other fusion strategies.
        """
        from sklearn.linear_model import LogisticRegression

        if self.detector_order is None:
            self.detector_order = sorted({name for ev in evidence_rows for name in ev})

        X = np.vstack([self._vectorize(ev) for ev in evidence_rows])
        y = np.asarray(y_true, dtype=int)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        if self.model is None:
            raise RuntimeError("LogisticMetaStrategy must be fit() before use.")
        x = self._vectorize(evidence).reshape(1, -1)
        proba = float(self.model.predict_proba(x)[0, 1])

        if proba >= self.anomaly_threshold:
            label = ANOMALY
        elif proba >= self.suspicious_threshold:
            label = SUSPICIOUS
        else:
            label = NORMAL

        return FusionResult(label=label, score=proba, strategy=self.name, contributing_detectors=evidence)

    def feature_names(self) -> List[str]:
        """Hand this to your XAI teammate along with `.model` for SHAP."""
        return list(self.detector_order or [])