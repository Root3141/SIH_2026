"""Turns raw detector output columns into normalized DetectorEvidence.

This (together with config.py) is the ONLY place that knows about
detector-specific column names. Strategies and the engine never touch a
raw column name directly.
"""
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from .base import DetectorEvidence
from .config import DetectorSpec


def row_to_evidence(row: pd.Series, spec: DetectorSpec) -> Optional[DetectorEvidence]:
    """Build DetectorEvidence for one detector from one row of a dataframe.

    Returns None if the detector's alert column is missing/NaN for this
    row, so strategies can gracefully handle a detector that hasn't run
    on part of the data (e.g. was added later, or failed on that row).
    """
    if spec.alert_col not in row.index or pd.isna(row.get(spec.alert_col)):
        return None

    alert = bool(row[spec.alert_col])

    if spec.score_col and spec.score_col in row.index and pd.notna(row[spec.score_col]):
        score = spec.normalize(float(row[spec.score_col]))
    else:
        # No continuous score available - fall back to binary alert.
        score = 1.0 if alert else 0.0

    severity = None
    if spec.severity_col and spec.severity_col in row.index and pd.notna(row[spec.severity_col]):
        severity = spec.severity_to_float(row[spec.severity_col])

    return DetectorEvidence(score=score, alert=alert, severity=severity)


def build_evidence_dict(row: pd.Series, specs: Dict[str, DetectorSpec]) -> Dict[str, DetectorEvidence]:
    """Build the full {detector_name: DetectorEvidence} dict for one row."""
    evidence = {}
    for name, spec in specs.items():
        ev = row_to_evidence(row, spec)
        if ev is not None:
            evidence[name] = ev
    return evidence