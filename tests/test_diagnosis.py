import json
import numpy as np
import pandas as pd
import pytest
from skyguard.diagnostics.diagnosis import DiagnosisReference, VARIABLES, OUTPUT_COLUMNS, diagnose


def reference(stations=('a',)):
    return DiagnosisReference(pd.DataFrame([dict(station_id=s,variable=v,residual_center=0.,residual_scale=1.,
        rate_center=0.,rate_scale=1.,rate_threshold=3.) for s in stations for v in VARIABLES]),'2025-12-31 23:00Z')


def frame(residual, index=-1):
    n=len(residual); peer=np.arange(n)*.2
    d=pd.DataFrame(dict(station_id='a',timestamp=pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC'),final_alert=False))
    for v in VARIABLES:
        d[v]=peer+20
        d['behavior_peer_expected_'+v]=peer+20
        d['behavior_peer_count_'+v]=3
    d.temperature+=np.array(residual)
    d.loc[d.index[index],'final_alert']=True
    return d


@pytest.mark.parametrize('residual,idx,expected',[
    ([0,0,0,0,0,np.nan],-1,'DATA_DROPOUT'),
    ([0,0,0,0,0,np.inf],-1,'DATA_DROPOUT'),
    ([0,0,10,0,0,0],2,'SPIKE'),
    ([6,6,6,6,6,6],-1,'SENSOR_OFFSET'),
    ([6,7,8,9,10,11],-1,'SENSOR_DRIFT'),
    ([-8,10,-15,7,-5,18],-1,'NOISY_SENSOR'),
    ([0,0,0,0,0,20],-1,'RATE_CHANGE'),
    ([0,0,0,0,0,0],-1,'UNCLASSIFIED_ANOMALY'),
])
def test_shapes(residual,idx,expected):
    d=frame(residual,idx)
    assert diagnose(d,reference()).diagnosis.iloc[idx]==expected


def test_stuck_and_precedence():
    d=frame([0]*6);d.temperature=20.
    assert diagnose(d,reference()).diagnosis.iloc[-1]=='STUCK_SENSOR'
    d.loc[5,'humidity']=np.nan
    assert diagnose(d,reference()).diagnosis.iloc[-1]=='DATA_DROPOUT'


def test_row_preservation_and_determinism():
    d=frame([6]*8).sample(frac=1,random_state=3);d.index=[4]*8
    d['statistical_alert']=True;d['temperature_spatial_residual']=8.
    before=d.copy(deep=True);a=diagnose(d,reference());b=diagnose(d,reference())
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    pd.testing.assert_frame_equal(d,before,check_exact=True)
    pd.testing.assert_frame_equal(a[d.columns],d,check_exact=True)
    for val in a.diagnosis_evidence:json.loads(val)


def test_no_ground_truth_or_counterfactual_leakage():
    d=frame([6]*8);baseline=diagnose(d,reference())[OUTPUT_COLUMNS]
    for c in ['is_anomaly','anomaly_type','anomaly_variable','anomaly_id','anomaly_event_id','synthetic_anomaly',
              'synthetic_anomaly_event_id','temperature_original','temperature_synthetic_anomaly_type']:
        d[c]=['random_'+str(i) for i in range(len(d))]
    pd.testing.assert_frame_equal(baseline,diagnose(d,reference())[OUTPUT_COLUMNS],check_exact=True)


def test_unalerted_not_diagnosed():
    d=frame([20]*8);d.final_alert=False
    assert diagnose(d,reference()).diagnosis.isna().all()


def test_gap_invalidates_sustained_shape():
    d=frame([6]*6);d.loc[3:,'timestamp']+=pd.Timedelta(hours=1)
    assert diagnose(d,reference()).diagnosis.iloc[-1]=='UNCLASSIFIED_ANOMALY'


def test_no_future_gap_confirmation():
    d=frame([0,0,10,0,0,0],2);d.loc[3:,'timestamp']+=pd.Timedelta(hours=1)
    assert diagnose(d,reference()).diagnosis.iloc[2]!='SPIKE'


def test_missing_peer_evidence_abstains():
    d=frame([6]*6);d=d[[c for c in d if not c.startswith('behavior_peer')]]
    assert diagnose(d,reference()).diagnosis.iloc[-1]=='UNCLASSIFIED_ANOMALY'


def test_history_separation():
    d=frame([0]*6);d.timestamp-=pd.Timedelta(days=365)
    with pytest.raises(ValueError,match='strictly precede'):diagnose(d,reference())


def test_station_independence_and_fault_corroboration():
    a=frame([6]*6);b=frame([0]*6);b.station_id='b';c=b.copy();c.station_id='c'
    d=pd.concat([a,b,c],ignore_index=True)
    result=diagnose(d,reference(('a','b','c')))
    assert result.operational_class.iloc[5]=='SENSOR_FAULT'
    assert result.diagnosis.iloc[11]=='UNCLASSIFIED_ANOMALY'
    assert diagnose(a,reference()).operational_class.iloc[5]=='UNCERTAIN'


def test_correlated_movement_is_only_possible_environment():
    frames=[]
    for s in ['a','b','c']:
        d=frame([0]*6);d.station_id=s
        for v in VARIABLES:
            d.loc[5,v]+=5;d.loc[5,'behavior_peer_expected_'+v]+=5
        frames.append(d)
    result=diagnose(pd.concat(frames,ignore_index=True),reference(('a','b','c')))
    assert result.loc[result.final_alert,'operational_class'].eq('POSSIBLE_ENVIRONMENTAL_EVENT').all()


def test_unavailable_window_does_not_make_stuck():
    d=frame([6]*6);d.loc[2,'temperature']=np.nan
    assert diagnose(d,reference()).diagnosis.iloc[-1]=='UNCLASSIFIED_ANOMALY'


def test_bad_keys_rejected():
    d=frame([0]*6);d.loc[1,'timestamp']=d.timestamp.iloc[0]
    with pytest.raises(ValueError,match='keys'):diagnose(d,reference())


def test_evaluation_alignment_defers_ground_truth_checks():
    from skyguard.evaluation.evaluate_diagnosis import align
    a=frame([0]*6);b=a.copy()
    for c in ['is_anomaly','anomaly_id','anomaly_type','anomaly_variable','anomaly_severity']:
        a[c]='original';b[c]='different'
    align(a,b,check_labels=False)
    with pytest.raises(ValueError,match='Benchmark mismatch'):align(a,b,check_labels=True)
