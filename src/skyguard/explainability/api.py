"""Operator-facing explanation contract with explicit boundaries."""
import json
import pandas as pd
from .alert_explanation import explain_decision
from .model_explanation import detector_evidence, unavailable, value
from .diagnosis_adapter import diagnose_fused

RECOMMENDATIONS = {
    'DATA_DROPOUT': 'Check telemetry delivery, power and sensor availability.',
    'STUCK_SENSOR': 'Inspect the sensor and compare with an independent reading.',
    'SENSOR_DRIFT': 'Review calibration history and compare with a reference instrument.',
    'SENSOR_OFFSET': 'Check calibration and installation against a reference instrument.',
    'NOISY_SENSOR': 'Inspect connections and interference; compare neighboring measurements.',
    'RATE_CHANGE': 'Review subsequent readings and nearby stations before assigning a fault.',
    'SPIKE': 'Review the transient and its recovery; inspect telemetry and local conditions.',
    'UNCLASSIFIED_ANOMALY': 'Review available readings and obtain additional evidence.',
}


def explain_alert(row, *, shap_context=None):
    decision=explain_decision(row)
    label=value(row,'diagnosis')
    evidence=value(row,'diagnosis_evidence')
    return dict(station_id=row['station_id'],timestamp=pd.Timestamp(row['timestamp']).isoformat(),
        final_alert=decision['final_alert'],decision=decision,detectors=detector_evidence(row),
        model_explanation=shap_context.explain(row) if shap_context else unavailable('not_requested'),
        diagnosis=dict(label=label,variable=value(row,'diagnosis_variable'),
            operational_class=value(row,'operational_class'),
            evidence=json.loads(evidence) if isinstance(evidence,str) else evidence,
            reason=value(row,'diagnosis_reason'),confidence_grade=value(row,'diagnosis_confidence'),
            confidence_is_probability=False,reference_status=value(row,'diagnosis_reference_status'),
            recommendation=RECOMMENDATIONS.get(label,'No operational diagnosis evaluated.')))


def explain_dataframe(fused, *, reference=None, shap_context=None, diagnose_rows=True):
    """Append objects positionally; preserve every existing column/index/order.

    For complete numeric evidence call enrich_evidence first. SHAP is opt-in and
    diagnosis is computed from its allowlisted raw/spatial fields, never labels.
    """
    if fused.duplicated(['station_id','timestamp']).any():
        raise ValueError('Duplicate station/timestamp keys')
    if 'explanation' in fused:
        raise ValueError('Explanation output already exists')
    source=diagnose_fused(fused,reference) if diagnose_rows and 'diagnosis' not in fused else fused
    result=source.copy()
    result['explanation']=[explain_alert(row,shap_context=shap_context) for row in source.to_dict('records')]
    return result
