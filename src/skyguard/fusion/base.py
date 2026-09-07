"""Core data structures for the SkyGuard fusion layer.

These abstractions exist so the fusion engine never needs to know
detector-specific column names or semantics. Anything detector-specific
lives in `adapters.py` / `config.py`; everything here is generic and
should never need to change when a detector is added or removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DetectorEvidence:
    """Normalized evidence produced by a single detector for one observation.

    score:    float in [0, 1]. Higher = more anomalous. If a detector has
              no continuous score, this falls back to the alert as 0/1.
    alert:    bool, the detector's own alert flag.
    severity: optional 0-1 severity value if the detector reports one
              (used by evidence-aware strategies; ignored by simple ones).
    """

    score: float
    alert: bool
    severity: Optional[float] = None


@dataclass
class FusionResult:
    """Output of a single fusion strategy for one observation."""

    alert: bool
    score: float
    strategy: str
    contributing_detectors: Dict[str, DetectorEvidence] = field(default_factory=dict)


class FusionStrategy:
    """Base class every fusion strategy must implement.

    A strategy receives a dict of {detector_name: DetectorEvidence} and must
    NOT assume which/how many detectors are present. This is what makes the
    engine modular: register a new detector in config.py, and every strategy
    that follows this contract keeps working without modification.
    """

    name: str = "base"

    def fuse(self, evidence: Dict[str, DetectorEvidence]) -> FusionResult:
        raise NotImplementedError