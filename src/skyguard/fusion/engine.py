"""Fusion engine: orchestrates adapters + strategies over a dataframe.

Adding a new detector to fusion never requires touching this file:
  1. Add a DetectorSpec to config.DETECTOR_REGISTRY (or register_detector()).
  2. Make sure its output columns exist in the dataframe you pass to run().
That's it - every registered strategy automatically sees the new evidence
the next time run() is called.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from .adapters import build_evidence_dict
from .base import FusionStrategy
from .config import DETECTOR_REGISTRY, DetectorSpec


class FusionEngine:
    def __init__(
        self,
        strategies: List[FusionStrategy],
        detector_specs: Optional[Dict[str, DetectorSpec]] = None,
    ):
        self.strategies = strategies
        self.detector_specs = detector_specs or DETECTOR_REGISTRY

    def evidence_for_row(self, row: pd.Series):
        return build_evidence_dict(row, self.detector_specs)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every registered strategy on every row of df.

        Adds two columns per strategy:
            fusion_<strategy>_label  (one of "normal"/"suspicious"/"anomaly")
            fusion_<strategy>_score  (continuous, for ranking/plotting)
        Leaves all original columns untouched.
        """
        out = df.copy()
        evidence_per_row = [self.evidence_for_row(row) for _, row in df.iterrows()]

        for strategy in self.strategies:
            labels, scores = [], []
            for evidence in evidence_per_row:
                result = strategy.fuse(evidence)
                labels.append(result.label)
                scores.append(result.score)
            out[f"fusion_{strategy.name}_label"] = labels
            out[f"fusion_{strategy.name}_score"] = scores

        return out