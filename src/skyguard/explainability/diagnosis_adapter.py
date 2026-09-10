"""Adapt current spatial fields to the validated diagnosis engine."""
import pandas as pd
from skyguard.diagnostics.diagnosis import DiagnosisReference, diagnose, OUTPUT_COLUMNS, VARIABLES


def diagnose_fused(fused, reference=None):
    allowed = ['station_id','timestamp','final_alert']+[v for v in VARIABLES if v in fused]
    data = fused[allowed].copy()
    for v in VARIABLES:
        for target, source in [('expected','expected'),('count','neighbor_count')]:
            key=f'{v}_spatial_{source}'
            if key in fused:
                data[f'behavior_peer_{target}_{v}']=fused[key].to_numpy()
    status='historical_reference_loaded'
    if reference is None:
        # No replacement normalizers or thresholds are estimated from evaluation.
        # Empty parameters trigger the engine's established abstention behavior;
        # raw nonfinite observations can still support DATA_DROPOUT.
        reference=DiagnosisReference(pd.DataFrame(columns=['station_id','variable','residual_center',
            'residual_scale','rate_center','rate_scale','rate_threshold']), '1900-01-01')
        status='historical_reference_unavailable; shape rules abstain'
    diagnosed=diagnose(data,reference)
    result=fused.copy()
    for column in OUTPUT_COLUMNS:
        result[column]=diagnosed[column].to_numpy()
    result['diagnosis_reference_status']=status
    return result
