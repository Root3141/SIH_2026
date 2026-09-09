from .base import (
    DetectorEvidence,
    FusionResult,
    FusionStrategy,
    NORMAL,
    SUSPICIOUS,
    ANOMALY,
    LABEL_ORDER,
)
from .config import DetectorSpec, DETECTOR_REGISTRY, register_detector
from .adapters import build_evidence_dict, row_to_evidence
from .strategies import CountTierStrategy, WeightedTierStrategy
from .meta_model import LogisticMetaStrategy
from .engine import FusionEngine

__all__ = [
    "DetectorEvidence",
    "FusionResult",
    "FusionStrategy",
    "NORMAL",
    "SUSPICIOUS",
    "ANOMALY",
    "LABEL_ORDER",
    "DetectorSpec",
    "DETECTOR_REGISTRY",
    "register_detector",
    "build_evidence_dict",
    "row_to_evidence",
    "CountTierStrategy",
    "WeightedTierStrategy",
    "LogisticMetaStrategy",
    "FusionEngine",
    "run_fusion",
    "assign_final_severity",
]


def __getattr__(name):
    if name in {"assign_final_severity", "run_fusion"}:
        from .fusion import assign_final_severity, run_fusion

        return {
            "assign_final_severity": assign_final_severity,
            "run_fusion": run_fusion,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
