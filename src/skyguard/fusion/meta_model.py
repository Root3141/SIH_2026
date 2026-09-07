"""Learned fusion: logistic regression over normalized detector scores.

This is the strategy to hand to your XAI teammate. Because it's a linear
model over a small number of *named* features (one per detector), SHAP
(or even just `model.coef_`) directly answers "how much did each detector
contribute to this fused decision" for any given row - no extra plumbing
needed on their end. Feature names == detector names, in a fixed order.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .base import DetectorEvidence, FusionResult, FusionStrategy


class LogisticMetaStrategy(FusionStrategy):
    name = "logistic_meta"

    def __init__(self, detector_order: Optional[List[str]] = None, threshold: float = 0.5):
        # detector_order fixes feature order for training/inference/SHAP.
        # If None, it's inferred from the training data on first fit().
        self.detector_order = detector_order
        self.threshold = threshold
        self.model = None

    def _vectorize(self, evidence: Dict[str, DetectorEvidence]) -> np.ndarray:
        order = self.detector_order or sorted(evidence.keys())
        return np.array([evidence[d].score if d in evidence else 0.0 for d in order])

    def fit(self, evidence_rows: List[Dict[str, DetectorEvidence]], y_true: List[bool]) -> None:
        """Train on rows of evidence dicts + ground-truth is_anomaly labels.

        Use rows from your shared synthetic benchmark (DEVELOPER_GUIDE
        section 20) so this is comparable to the other fusion baselines.
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
        return FusionResult(
            alert=proba >= self.threshold,
            score=proba,
            strategy=self.name,
            contributing_detectors=evidence,
        )

    def feature_names(self) -> List[str]:
        """Hand this to your XAI teammate along with `.model` for SHAP."""
        return list(self.detector_order or [])