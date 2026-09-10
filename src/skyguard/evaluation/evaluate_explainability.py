"""Evaluate explanation integrity against current fusion, without SHAP accuracy claims."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from skyguard.config.paths import PROJECT_ROOT, HISTORICAL_PARQUET
from skyguard.fusion.fusion import run_fusion, _load_artifacts, ARTIFACTS_DIR
from skyguard.explainability import enrich_evidence, explain_dataframe, ReconstructionShap, ShapConfig
from skyguard.evaluation.evaluate_diagnosis import load_reference, evaluate_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=PROJECT_ROOT/'data/synthetic/skyguard_demo_2026.parquet')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/explainability')
    parser.add_argument('--artifacts-dir',type=Path,default=ARTIFACTS_DIR)
    parser.add_argument('--background',type=Path,default=HISTORICAL_PARQUET)
    parser.add_argument('--max-shap-rows',type=int,default=2)
    parser.add_argument('--background-size',type=int,default=25)
    parser.add_argument('--shap-seconds',type=float,default=30.)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--historical-end')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    raw=pd.read_parquet(args.input)
    runtime=raw[[c for c in ['station_id','timestamp','temperature','pressure','humidity','surface_pressure'] if c in raw]].copy()
    print('Running unchanged frozen fusion...',flush=True)
    fused=run_fusion(runtime,args.artifacts_dir)
    print('Recovering detector evidence...',flush=True)
    enriched=enrich_evidence(fused,args.artifacts_dir)
    calibration=_load_artifacts(args.artifacts_dir)[2]
    # Background is the chronological training portion used by calibration,
    # excluding its held-out validation period. No labels are selected.
    background=pd.read_parquet(args.background,columns=['station_id','timestamp',*calibration.variables])
    cutoff=background.timestamp.min()+(background.timestamp.max()-background.timestamp.min())*(1-calibration.config.val_fraction)
    background=background[background.timestamp<cutoff]
    config=ShapConfig(max_rows=args.max_shap_rows,background_size=args.background_size,max_seconds=args.shap_seconds)
    context=ReconstructionShap(runtime,calibration,background,config)
    print('Generating explanations and operational diagnoses...',flush=True)
    result=explain_dataframe(enriched,reference=load_reference(args.reference,args.historical_end),shap_context=context)
    pd.testing.assert_frame_equal(result[fused.columns],fused,check_exact=True)
    # Re-run the explanation/diagnosis boundary with adversarial labels; no SHAP
    # recomputation is needed because its context selects raw columns explicitly.
    poison=enriched.assign(is_anomaly=False,anomaly_type='poison',anomaly_variable='poison',event_id='poison',temperature_original=-999.)
    check=explain_dataframe(poison,reference=load_reference(args.reference,args.historical_end))
    for expected,actual in zip(result.explanation,check.explanation):
        assert {k:v for k,v in expected.items() if k!='model_explanation'}=={k:v for k,v in actual.items() if k!='model_explanation'}
    explanations=result.explanation.tolist()
    shap_results=[e['model_explanation'] for e in explanations]
    successful=[s for s in shap_results if s['available']]
    alerts=result.final_alert
    complete=sum(all(d['available'] for d in e['detectors'].values()) for e in explanations)
    assigned=int(result.loc[alerts,'diagnosis'].notna().sum())
    specific=int(result.loc[alerts,'diagnosis'].ne('UNCLASSIFIED_ANOMALY').sum())
    metrics=dict(fused_rows=len(fused),final_alerts=int(alerts.sum()),explanations_generated=len(explanations),
        complete_detector_explanations=complete,
        lstm_reconstruction_explanations=sum(e['detectors']['lstm']['available'] for e in explanations),
        shap_explanations_generated=len(successful),shap_unavailable=len(shap_results)-len(successful),
        shap_attempts=context.attempts,shap_status_counts=dict(Counter(s.get('reason','available') for s in shap_results)),
        average_shap_runtime_seconds=float(np.mean([s['runtime_seconds'] for s in successful])) if successful else None,
        diagnosis_assigned_alerts=assigned,diagnosis_specific_alerts=specific,
        diagnosis_coverage=assigned/int(alerts.sum()) if alerts.any() else 0.,
        specific_diagnosis_coverage=specific/int(alerts.sum()) if alerts.any() else 0.,
        diagnosis_reference_status=result.diagnosis_reference_status.iloc[0],
        no_row_loss=len(result)==len(fused),no_duplicate_keys=not result.duplicated(['station_id','timestamp']).any(),
        protected_columns_unchanged=True,ground_truth_independence=True,
        total_runtime_seconds=perf_counter()-started,shap_background_size=config.background_size,
        backend=calibration.backend,model_class=type(calibration.model_state).__name__)
    result.drop(columns='explanation').to_parquet(args.output/'explained_rows.parquet',index=False)
    operator=result[['station_id','timestamp']].copy()
    operator['explanation']=[json.dumps(e,allow_nan=False) for e in explanations]
    operator.to_parquet(args.output/'operator_explanations.parquet',index=False)
    # Operator sidecar contains only the structured allowlisted objects.
    with (args.output/'explanations.jsonl').open('w') as handle:
        for e in explanations:handle.write(json.dumps(e,allow_nan=False)+'\n')
    alert_examples=[e for e in explanations if e['final_alert']]
    sample=next((e for e in alert_examples if e['model_explanation']['available']),alert_examples[0] if alert_examples else None)
    (args.output/'sample_alert.json').write_text(json.dumps(sample,indent=2,allow_nan=False))
    metrics['diagnosis_evaluation']=evaluate_predictions(result.drop(columns='explanation'),raw,args.output/'diagnosis')
    (args.output/'integrity_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
