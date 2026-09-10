"""Phase-1 diagnosis evaluation on current frozen 2-of-3 fusion output.

Predictions precede all label joins. Missing historical references cause explicit
abstention; no diagnosis normalizers, rules or thresholds are fitted here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from skyguard.config.paths import RESULTS_DIR, PROJECT_ROOT
from skyguard.diagnostics.diagnosis import DiagnosisReference, VARIABLES
from skyguard.explainability.diagnosis_adapter import diagnose_fused
from skyguard.fusion.fusion import run_fusion, ARTIFACTS_DIR

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
    columns=[v for v in VARIABLES if v in left]+(['is_anomaly','anomaly_id','anomaly_type','anomaly_variable','anomaly_severity'] if check_labels else [])
    for c in columns:
        a,b=left[c].reset_index(drop=True),right[c]
        if not (a.eq(b)|(a.isna()&b.isna())).all():raise ValueError('Benchmark mismatch: '+c)
    return right


def classification(truth,pred,labels):
    p,r,f,s=precision_recall_fscore_support(truth,pred,labels=labels,zero_division=0)
    per=pd.DataFrame(dict(diagnosis=labels,precision=p,recall=r,f1=f,support=s))
    summary=dict(rows=len(truth),accuracy=accuracy_score(truth,pred) if len(truth) else None,
        macro_f1=float(np.mean(f)),weighted_f1=float(np.dot(f,s)/s.sum()) if s.sum() else None)
    matrix=confusion_matrix(truth,pred,labels=labels+['UNCLASSIFIED_ANOMALY']) if len(truth) else np.zeros((len(labels)+1,len(labels)+1),dtype=int)
    return per,summary,pd.DataFrame(matrix,index=labels+['UNCLASSIFIED_ANOMALY'],columns=labels+['UNCLASSIFIED_ANOMALY'])



def load_reference(path=None, historical_end=None):
    if path is None:
        return None
    if historical_end is None:
        raise ValueError('--historical-end is required with --reference')
    return DiagnosisReference(pd.read_csv(path), historical_end)


def evaluate_predictions(predictions, truth_frame, output=OUTPUT):
    """Only call after diagnosis; synthetic columns are joined here for scoring."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    keys=['station_id','timestamp']
    labels=['is_anomaly','anomaly_type','anomaly_variable','anomaly_id']
    if not set(labels[:3]).issubset(truth_frame):
        return {'status':'synthetic_ground_truth_unavailable'}
    truth=truth_frame[keys+[c for c in labels if c in truth_frame]].copy()
    truth.timestamp=pd.to_datetime(truth.timestamp,utc=True)
    result=predictions.merge(truth,on=keys,validate='one_to_one')
    if len(result)!=len(predictions):raise ValueError('Evaluation row loss')
    eligible=result.final_alert & result.is_anomaly
    single=eligible & ~result.anomaly_type.fillna('').str.contains('|',regex=False)
    target=result.loc[single,'anomaly_type'].map(LABELS)
    if target.isna().any():raise ValueError('Unsupported benchmark phenotype')
    pred=result.loc[single,'diagnosis']
    per,summary,matrix=classification(target,pred,list(LABELS.values()))
    summary.update(total_rows=len(result),upstream_alert_rows=int(result.final_alert.sum()),
        detected_anomalous_rows=int(eligible.sum()),
        upstream_missed_anomaly_rows=int((result.is_anomaly & ~result.final_alert).sum()),
        upstream_false_positive_rows=int((result.final_alert & ~result.is_anomaly).sum()),
        diagnosis_coverage=float(result.loc[eligible,'diagnosis'].ne('UNCLASSIFIED_ANOMALY').mean()) if eligible.any() else 0.,
        overlapping_detected_rows=int((eligible & ~single).sum()),
        joint_shape_variable_accuracy=float((pred.eq(target)&result.loc[single,'diagnosis_variable'].eq(result.loc[single,'anomaly_variable'])).mean()) if single.any() else None)
    family_per,family_summary,family_matrix=classification(target.map(FAMILIES),pred.map(FAMILIES),list(dict.fromkeys(FAMILIES[x] for x in LABELS.values())))
    per.to_csv(output/'diagnosis_by_class.csv',index=False)
    matrix.to_csv(output/'diagnosis_confusion_matrix.csv')
    family_per.to_csv(output/'diagnosis_family_by_class.csv',index=False)
    family_matrix.to_csv(output/'diagnosis_family_confusion_matrix.csv')
    # Preserve overlap-inclusive event membership analysis without the missing old module.
    members=[]
    if 'anomaly_id' in result:
        for r in result.loc[result.is_anomaly].to_dict('records'):
            ids=str(r['anomaly_id']).split('|'); kinds=str(r['anomaly_type']).split('|')
            variables=str(r['anomaly_variable']).split('|')
            if not (len(ids)==len(kinds)==len(variables)):
                raise ValueError('Misaligned event membership labels')
            for event,kind,variable in zip(ids,kinds,variables):
                members.append(dict(anomaly_id=event,anomaly_type=kind,anomaly_variable=variable,
                                    final_alert=r['final_alert'],diagnosis=r['diagnosis']))
    if members:
        expanded=pd.DataFrame(members); records=[]
        for event,g in expanded.groupby('anomaly_id'):
            detected=g[g.final_alert];counts=detected.diagnosis.value_counts()
            dominant=sorted(counts[counts.eq(counts.max())].index)[0] if len(counts) else None
            records.append(dict(anomaly_id=event,total_event_rows=len(g),detected_event_rows=len(detected),
                dominant_diagnosis=dominant,conflicting_diagnoses=len(counts)>1,
                dominant_percent_detected=100*counts.max()/len(detected) if len(detected) else None))
        pd.DataFrame(records).to_csv(output/'diagnosis_event_consistency.csv',index=False)
        detected=expanded[expanded.final_alert]
        member_per,member_summary,_=classification(detected.anomaly_type.map(LABELS),detected.diagnosis,list(LABELS.values()))
        member_per.to_csv(output/'diagnosis_membership_by_class.csv',index=False)
        (output/'diagnosis_membership_metrics.json').write_text(json.dumps(member_summary,indent=2))
    (output/'diagnosis_metrics.json').write_text(json.dumps(summary,indent=2))
    (output/'diagnosis_family_metrics.json').write_text(json.dumps(family_summary,indent=2))
    result.loc[result.final_alert & ~result.is_anomaly].groupby('diagnosis').size().to_csv(output/'upstream_false_positive_diagnoses.csv')
    result.loc[result.final_alert].groupby('operational_class').size().to_csv(output/'operational_classes.csv')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=PROJECT_ROOT/'data/synthetic/skyguard_demo_2026.parquet')
    parser.add_argument('--artifacts-dir',type=Path,default=ARTIFACTS_DIR)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--historical-end')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    raw=pd.read_parquet(args.input)
    runtime=raw[['station_id','timestamp']+[v for v in VARIABLES if v in raw]].copy()
    fused=run_fusion(runtime,args.artifacts_dir)
    predictions=diagnose_fused(fused,load_reference(args.reference,args.historical_end))
    pd.testing.assert_frame_equal(predictions[fused.columns],fused,check_exact=True)
    args.output.mkdir(parents=True,exist_ok=True)
    predictions.to_parquet(args.output/'diagnosed_evaluation.parquet',index=False)
    print(json.dumps(evaluate_predictions(predictions,raw,args.output),indent=2))
    print(predictions.diagnosis_reference_status.iloc[0])


if __name__=='__main__':main()
