"""Post-detection explanations; never part of the alert decision."""
from .alert_explanation import explain_decision
from .api import explain_alert, explain_dataframe
from .evidence import enrich_evidence
from .model_explanation import ReconstructionShap, ShapConfig
