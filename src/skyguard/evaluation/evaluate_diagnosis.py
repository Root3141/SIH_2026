"""Phase-1 evaluation conditional on frozen upstream detection; no model operations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from skyguard.config.paths import RESULTS_DIR, CANONICAL_EVAL_INJECTED_PARQUET
from skyguard.diagnostics.diagnosis import DiagnosisReference, diagnose, VARIABLES, PRECEDENCE, OUTPUT_COLUMNS
from skyguard.evaluation.evaluate_complementarity import event_rows

OUTPUT=RESULTS_DIR/'diagnosis'
LABELS=dict(dropout='DATA_DROPOUT',stuck='STUCK_SENSOR',drift='SENSOR_DRIFT',offset='SENSOR_OFFSET',
            noise='NOISY_SENSOR',rate_change='RATE_CHANGE',spike='SPIKE')
FAMILIES=dict(DATA_DROPOUT='DATA_AVAILABILITY',STUCK_SENSOR='FIXED_VALUE_FAILURE',
    SENSOR_DRIFT='BIAS_FAILURE',SENSOR_OFFSET='BIAS_FAILURE',NOISY_SENSOR='SIGNAL_INSTABILITY',
    RATE_CHANGE='SIGNAL_INSTABILITY',SPIKE='TRANSIENT_EXCURSION',UNCLASSIFIED_ANOMALY='UNCLASSIFIED_ANOMALY')


def align(reference,other,check_labels=True):
    keys=['station_id','timestamp']
    right=other.copy();right.timestamp=pd.to_datetime(right.timestamp,utc=True)
    left=reference.copy();left.timestamp=pd.to_datetime(left.timestamp,utc=True)
    for frame in [left,right]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any(): raise ValueError('Invalid cached keys')
    idx=pd.MultiIndex.from_frame(left[keys]);right=right.set_index(keys)
    if set(idx)!=set(right.index):raise ValueError('Benchmark key mismatch')
    right=right.loc[idx].reset_index()
    columns=VARIABLES+(['is_anomaly','anomaly_id','anomaly_type','anomaly_variable','anomaly_severity'] if check_labels else [])
    for c in columns:
        a,b=left[c].reset_index(drop=True),right[c]
        if not (a.eq(b)|(a.isna()&b.isna())).all():raise ValueError('Benchmark mismatch: '+c)
    return right


def classification(truth,pred,labels):
    p,r,f,s=precision_recall_fscore_support(truth,pred,labels=labels,zero_division=0)
    per=pd.DataFrame(dict(diagnosis=labels,precision=p,recall=r,f1=f,support=s))
    summary=dict(rows=len(truth),accuracy=accuracy_score(truth,pred) if len(truth) else np.nan,
        macro_f1=float(np.mean(f)),weighted_f1=float(np.dot(f,s)/s.sum()) if s.sum() else np.nan)
    matrix=confusion_matrix(truth,pred,labels=labels+['UNCLASSIFIED_ANOMALY'])
    return per,summary,pd.DataFrame(matrix,index=labels+['UNCLASSIFIED_ANOMALY'],columns=labels+['UNCLASSIFIED_ANOMALY'])


def verify_protected():
    before=json.loads((OUTPUT/'preservation_before.json').read_text())
    after={p:hashlib.sha256((RESULTS_DIR.parent/p).read_bytes()).hexdigest() for p in before}
    if before!=after:raise ValueError('Protected artifacts or source changed')
    (OUTPUT/'preservation_after.json').write_text(json.dumps(after,indent=2))
    return len(after)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    canonical=pd.read_parquet(CANONICAL_EVAL_INJECTED_PARQUET)
    original=pd.read_parquet(RESULTS_DIR/'final_pipeline/final_predictions.parquet')
    align(canonical,original,check_labels=False)
    behavior=align(original,pd.read_parquet(RESULTS_DIR/'behavior_phase2/full_results.parquet'),check_labels=False)
    data=original.copy()
    for v in VARIABLES:
        for kind in ['expected','count']:
            c=f'behavior_peer_{kind}_{v}';data[c]=behavior[c].to_numpy()
    metadata=json.loads((RESULTS_DIR/'behavior_phase2/calibration_metadata.json').read_text())
    reference=DiagnosisReference(pd.read_csv(RESULTS_DIR/'behavior_phase2/threshold_metadata.csv'),metadata['historical_end'])
    # Predictions are completed before ANY evaluation-label filtering or scoring.
    result=diagnose(data,reference)
    # Labels may now be validated for evaluation; they never governed prediction.
    align(canonical,original)
    align(original,behavior)
    pd.testing.assert_frame_equal(result[original.columns],original,check_exact=True)
    result.to_parquet(OUTPUT/'diagnosed_evaluation.parquet',index=False)
    eligible=result.final_alert & result.is_anomaly
    single=eligible & ~result.anomaly_type.fillna('').str.contains('|',regex=False)
    truth=result.loc[single,'anomaly_type'].map(LABELS)
    if truth.isna().any():raise ValueError('Unsupported benchmark phenotype')
    pred=result.loc[single,'diagnosis']
    per,summary,matrix=classification(truth,pred,list(LABELS.values()))
    summary.update(total_rows=len(result),upstream_alert_rows=int(result.final_alert.sum()),
        detected_anomalous_rows=int(eligible.sum()),upstream_missed_anomaly_rows=int((result.is_anomaly&~result.final_alert).sum()),
        upstream_false_positive_rows=int((result.final_alert&~result.is_anomaly).sum()),
        specific_diagnoses=int((eligible & result.diagnosis.ne('UNCLASSIFIED_ANOMALY')).sum()),
        diagnosis_coverage=float(result.loc[eligible,'diagnosis'].ne('UNCLASSIFIED_ANOMALY').mean()),
        overlapping_detected_rows=int((eligible&~single).sum()),
        joint_shape_variable_accuracy=float((pred.eq(truth)&result.loc[single,'diagnosis_variable'].eq(result.loc[single,'anomaly_variable'])).mean()))
    family_per,family_summary,family_matrix=classification(truth.map(FAMILIES),pred.map(FAMILIES),list(dict.fromkeys(FAMILIES[x] for x in LABELS.values())))
    expanded=event_rows(result)
    records=[]
    for event,g in expanded.groupby('anomaly_id'):
        detected=g[g.final_alert]
        counts=detected.diagnosis.value_counts()
        dominant=sorted(counts[counts.eq(counts.max())].index)[0] if len(counts) else None
        records.append(dict(anomaly_id=event,anomaly_type=g.anomaly_type.iloc[0],anomaly_variable=g.anomaly_variable.iloc[0],
            total_event_rows=len(g),detected_event_rows=len(detected),dominant_diagnosis=dominant,
            dominant_percent_detected=100*counts.max()/len(detected) if len(detected) else np.nan,
            dominant_percent_all_event_rows=100*counts.max()/len(g) if len(detected) else np.nan,
            conflicting_diagnoses=len(counts)>1,distinct_diagnoses=len(counts)))
    events=pd.DataFrame(records)
    # Secondary overlap-inclusive membership analysis: each constituent event
    # membership receives the same row diagnosis; no oracle chooses a label.
    members=expanded[expanded.final_alert]
    overlap_per,overlap_summary,_=classification(members.anomaly_type.map(LABELS),members.diagnosis,list(LABELS.values()))
    tables=dict(diagnosis_metrics=pd.DataFrame([summary]),diagnosis_by_class=per,
        diagnosis_family_metrics=pd.DataFrame([family_summary]),diagnosis_family_by_class=family_per,
        diagnosis_event_consistency=events,diagnosis_membership_metrics=pd.DataFrame([overlap_summary]),
        diagnosis_membership_by_class=overlap_per,
        upstream_false_positive_diagnoses=result.loc[result.final_alert&~result.is_anomaly].groupby('diagnosis').size().reset_index(name='rows'),
        operational_classes=result.loc[result.final_alert].groupby('operational_class').size().reset_index(name='rows'))
    for name,table in tables.items():table.to_csv(OUTPUT/(name+'.csv'),index=False)
    matrix.to_csv(OUTPUT/'diagnosis_confusion_matrix.csv',index_label='true_diagnosis')
    family_matrix.to_csv(OUTPUT/'diagnosis_family_confusion_matrix.csv',index_label='true_family')
    source=dict(reference_path=str(RESULTS_DIR/'behavior_phase2/threshold_metadata.csv'),metadata=metadata,
        rule_origin='Fixed engineering rules, declared before evaluation; no label-based tuning',precedence=PRECEDENCE)
    (OUTPUT/'reference_metadata.json').write_text(json.dumps(source,indent=2))
    protected=verify_protected()
    report=['# Sensor Health & Anomaly Diagnosis — Phase 1',
        'Post-detection, retrospective batch diagnosis. No new detector, model training, alert modifications or fusion integration.',
        '## Evaluation contract',
        'Eligibility is the existing frozen production final_alert. All rows (including unalerted context) are preserved; '
        'nonalerts have null diagnosis. Predictions are made from an explicit allowlist before reading evaluation labels. '
        'Primary class metrics use singly labeled detected anomalous rows; overlapping rows are explicitly excluded from that '
        'confusion matrix and separately reported in overlap-inclusive event-membership metrics. '
        'Unknown diagnoses count as errors/false negatives. Macro F1 averages seven true phenotype classes, excluding the abstention output class. '
        'Joint shape-variable accuracy additionally requires the chosen variable to match truth. '
        'Upstream false positives have no valid fault-shape truth; their diagnoses are reported separately. '
        'This conditional diagnosis accuracy is not end-to-end detection recall and cannot credit missed anomalies.',
        'Fault-family mapping: DATA_AVAILABILITY = dropout; FIXED_VALUE_FAILURE = stuck; '
        'BIAS_FAILURE = drift + offset; SIGNAL_INSTABILITY = noise + rate_change; '
        'TRANSIENT_EXCURSION = spike. These group availability loss, motion collapse, persistent bias, '
        'unstable signal changes and isolated excursions respectively. Ramp-generating drift/rate-change '
        'injections cross this operational naming boundary, so family accuracy is not a physical identifiability guarantee.',
        '## Schema audit',
        'Production final_alert is Statistical OR Spatial OR Missingness. Missingness checks four variables; numeric core uses '
        'temperature/humidity/pressure and Spatial excludes humidity from its final vote. Statistical supplies hourly residuals, '
        'ROC and rolling z-scores. Spatial supplies IDW expectations, signed residuals and neighbor counts. '
        'Phase-2 behavior exposes matching IDW peer expectations and historical robust normalization; these are used instead of '
        'mapping detector flags to fault labels. TCN/GRU forecast and residual outputs exist but are not required. LSTM outputs are absent.',
        '## Fixed rules and precedence',
        'Six trailing complete hourly observations are required for sustained shapes. Gaps and nonfinite evidence invalidate the window. '
        'Order: DATA_DROPOUT (nonfinite raw value), STUCK_SENSOR (raw span <=0.05 native units and peer span >=0.2 and >=4 times raw span), '
        'NOISY_SENSOR (MAD of standardized residual derivatives >=3 and reversal fraction >=0.6), '
        'SPIKE (absolute normalized residual >=3, preceding two residuals and following residual <2), '
        'SENSOR_DRIFT (same-sign fraction >=0.8, absolute median residual >=3, absolute median slope >0.25 normalized units/hour '
        'and monotonic fraction >=0.8), SENSOR_OFFSET (same sustained bias, slope <=0.25 and residual MAD <=1), '
        'RATE_CHANGE (absolute standardized derivative >= frozen historical rate threshold), then UNCLASSIFIED_ANOMALY. '
        'Availability precedes numerical rules; motion collapse precedes bias; erratic movement precedes apparent bias; '
        'confirmed transient recovery precedes sustained-bias attribution; derivative-only evidence is least specific. '
        'Across variables, precedence wins, then absolute normalized residual, then fixed variable order. '
        'Constants are engineering definitions, not benchmark estimates: 0.05 reuses repository motion resolution; six samples '
        'require sustained evidence; 3/2 robust-scale separation distinguishes large and quiet residuals; 0.8 denotes near-consistent '
        'sign, 0.6 frequent reversal, 0.25 slow-versus-stable change, and a 4:1 peer motion contrast avoids ordinary quiet weather. '
        'These choices are provisional, not empirically calibrated class probabilities.',
        '## Timing, confidence and operational class',
        'Spike confirmation requires exactly one following hourly observation. It cannot be emitted at an excursion onset in a '
        'causal live pipeline; boundary excursions abstain or receive another supported diagnosis. Other shapes use current/past data. '
        'Confidence is a fixed evidence grade (0.95 missing, 0.85 stuck, 0.8 spike, 0.7 noise, 0.65 bias, 0.55 rate, 0 abstention), '
        'not a probability. SENSOR_FAULT requires local absolute normalized residual >=3, >=2 spatial peers and >=2 other network '
        'stations with residual <2. POSSIBLE_ENVIRONMENTAL_EVENT requires similar contemporaneous movement by >=2 other stations '
        'and the peer reference, with a small local residual and raw motion >=3 historical rate scales; '
        'otherwise UNCERTAIN. These are corroboration hypotheses, not causal proof. '
        'No regional-event ground truth exists here, so operational class accuracy is not claimed.',
        '## Calibration source','```json\n'+json.dumps(source,indent=2)+'\n```',
        '## Results','```json\n'+json.dumps(summary,indent=2)+'\n```']
    for name in ['diagnosis_by_class','diagnosis_family_metrics','diagnosis_family_by_class','diagnosis_membership_metrics','operational_classes']:
        report.extend(['## '+name,'```text\n'+tables[name].to_string(index=False)+'\n```'])
    report.extend(['## Event consistency',f'{int(events.detected_event_rows.gt(0).sum())}/{len(events)} events detected upstream; '
        f'{int(events.conflicting_diagnoses.sum())} detected events have conflicting diagnoses (including abstention). '
        'Dominance ties use lexical order. Both detected-observation and whole-event denominators are exported.',
        '## Limitations','Single-benchmark exploratory evidence. Drift and injected rate-change are both additive ramps and can be '
        'observationally indistinguishable. Offset vs drift is sensitive to weather variation, window length and onset. '
        'Clipping can turn excursions into plateaus. Multi-variable rows are reduced to a single primary diagnosis. '
        'Peer contamination, natural station differences and historical normalization shifts can mislead localization. '
        'The 0.05 resolution is inherited but is not a verified hardware specification. No rules were tuned after evaluation.',
        '## Preservation',f'All {protected} protected source/data/result files match their initial SHA-256 hashes. '
        'Existing full-suite synthetic training fixtures are separate from this engine; no model is trained by diagnosis.',
        '## Reproduce','PYTHONPATH=src .venv/bin/python -m skyguard.evaluation.evaluate_diagnosis'])
    for filename,heading in [('verdict.md','Verdict'),('test_verification.txt','Tests')]:
        if (OUTPUT/filename).exists():report.extend(['## '+heading,(OUTPUT/filename).read_text()])
    (OUTPUT/'diagnosis_report.md').write_text('\n\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2));print(per.to_string(index=False));print(family_summary)


if __name__=='__main__':main()
