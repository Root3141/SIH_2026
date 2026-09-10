import numpy as np
import pandas as pd
import pytest
from skyguard.explainability import explain_decision


def row(n=2):
    return dict(station_id='AWS_001', timestamp=pd.Timestamp('2026-01-02', tz='UTC'),
                statistical_alert=n>=1, spatial_alert=n>=2, lstm_ae_alert=n>=3,
                detector_votes=n, final_alert=n>=2, final_severity=['normal','suspicious','anomaly','critical'][n],
                final_confidence=n/3)


@pytest.mark.parametrize('n', range(4))
def test_votes(n):
    result=explain_decision(row(n))
    assert result['agreement_count']==n
    assert result['final_alert']==(n>=2)
    assert ('no final anomaly alert was raised' in result['alert_explanation_summary']) == (n<2)


@pytest.mark.parametrize('field,value', [('final_alert',False),('detector_votes',1),('statistical_alert',np.nan)])
def test_inconsistent_decision(field,value):
    d=row();d[field]=value
    with pytest.raises(ValueError):explain_decision(d)


def test_decision_label_independence():
    d=row(); expected=explain_decision(d)
    d.update(is_anomaly=False, anomaly_type='spike', anomaly_variable='pressure', event_id='fake')
    assert explain_decision(d)==expected

from skyguard.explainability.model_explanation import detector_evidence

@pytest.mark.parametrize('name',['statistical','spatial','lstm'])
def test_missing_optional_evidence(name):
    assert not detector_evidence(row())[name]['available']


def test_reconstruction_existing_fields():
    d=row(); d.update(lstm_ae_top_contributor='humidity',humidity=80.,humidity_lstm_ae_error=4.,
                       humidity_lstm_ae_expected=60.,humidity_lstm_ae_residual=20.)
    e=detector_evidence(d)['lstm']
    assert e['primary_variable']=='humidity'
    assert e['evidence']['humidity']==dict(observed=80.,error=4.,expected=60.,residual=20.)

from skyguard.explainability.model_explanation import ReconstructionShap, ShapConfig, ObservationScore, normalized_windows
from skyguard.fusion.fusion import _load_artifacts, ARTIFACTS_DIR
from skyguard.detectors.lstm_autoencoder import run_lstm_autoencoder_detector


def weather(n=50):
    return pd.DataFrame(dict(station_id='AWS_001',timestamp=pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC'),
                             temperature=20+np.sin(np.arange(n)),pressure=1010.,humidity=50.))


@pytest.fixture
def calibration():
    return _load_artifacts(ARTIFACTS_DIR)[2]


def test_exact_observation_wrapper(calibration):
    d=weather();result=run_lstm_autoencoder_detector(d,calibration)
    data,norm,windows,indices=normalized_windows(d,calibration)
    for target in [0,23,49]:
        selected=indices[np.any(indices==target,axis=1)];context=np.unique(selected)
        score=ObservationScore(calibration,context,selected,target,ShapConfig())
        assert score(norm.loc[context,calibration.variables].to_numpy().reshape(1,-1))[0]==pytest.approx(result.lstm_ae_score.iloc[target],rel=1e-12)


def test_insufficient_history(calibration):
    d=weather(2);r=row();r['timestamp']=d.timestamp.iloc[0];r['lstm_ae_score']=1.
    output=ReconstructionShap(d,calibration,d).explain(r)
    assert not output['shap_available']
    assert 'insufficient_history' in output['reason']


def test_shap_import_failure(calibration,monkeypatch):
    import sys
    d=weather();r=run_lstm_autoencoder_detector(d,calibration).iloc[24].to_dict();r['final_alert']=True
    bg=d.copy();bg.timestamp-=pd.Timedelta(days=365)
    monkeypatch.setitem(sys.modules,'shap',None)
    output=ReconstructionShap(d,calibration,bg).explain(r)
    assert not output['shap_available']
    assert 'ModuleNotFoundError' in output['reason']


def test_shap_budget(calibration):
    d=weather();r=row()
    assert ReconstructionShap(d,calibration,d,ShapConfig(max_rows=0)).explain(r)['reason']=='row_budget_exceeded'

from skyguard.explainability import explain_alert, explain_dataframe, enrich_evidence
from skyguard.explainability.diagnosis_adapter import diagnose_fused
from skyguard.fusion.fusion import run_fusion


def test_dataframe_preservation_and_label_independence():
    d=pd.DataFrame([dict(row(n),timestamp=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(hours=n)) for n in range(4)])
    d=d.iloc[::-1].copy();d.index=[4]*4
    before=d.copy(deep=True)
    result=explain_dataframe(d,diagnose_rows=False)
    pd.testing.assert_frame_equal(result[d.columns],before,check_exact=True)
    assert len(result)==4 and not result.duplicated(['station_id','timestamp']).any()
    for r,e in zip(d.to_dict('records'),result.explanation):
        assert e['station_id']==r['station_id']
        assert pd.Timestamp(e['timestamp'])==r['timestamp']
    for c in ['is_anomaly','anomaly_type','anomaly_variable','event_id','temperature_original']:
        d[c]='poison'
    assert result.explanation.tolist()==explain_dataframe(d,diagnose_rows=False).explanation.tolist()


def test_actual_fusion_evidence_preservation():
    d=weather(48)
    d.loc[24,'temperature']=80.
    fused=run_fusion(d);before=fused.copy(deep=True)
    enriched=enrich_evidence(fused)
    result=explain_dataframe(enriched)
    pd.testing.assert_frame_equal(result[fused.columns],before,check_exact=True)
    pd.testing.assert_frame_equal(fused,before,check_exact=True)
    assert result.explanation.iloc[24]['detectors']['statistical']['available']
    assert result.explanation.iloc[24]['detectors']['lstm']['available']
    polluted=fused.assign(is_anomaly=False,anomaly_type='fake',event_id='fake',temperature_original=-999.)
    assert explain_dataframe(enrich_evidence(polluted)).explanation.tolist()==result.explanation.tolist()


def test_current_three_variable_diagnosis_no_fake_dropout():
    d=weather(6);d['final_alert']=True
    result=diagnose_fused(d)
    assert result.diagnosis.eq('UNCLASSIFIED_ANOMALY').all()
    assert result.diagnosis_reference_status.str.contains('unavailable').all()


def test_duplicate_keys_refused():
    d=pd.DataFrame([row(),row()])
    with pytest.raises(ValueError,match='Duplicate'):explain_dataframe(d,diagnose_rows=False)


def test_prediction_allowlist(monkeypatch):
    import skyguard.explainability.diagnosis_adapter as adapter
    original=adapter.diagnose
    def spy(data,reference):
        assert not any(c in data for c in ['is_anomaly','anomaly_type','event_id','temperature_original'])
        return original(data,reference)
    monkeypatch.setattr(adapter,'diagnose',spy)
    d=weather(6).assign(final_alert=True,is_anomaly=True,anomaly_type='spike',event_id='fake',temperature_original=3.)
    diagnose_fused(d)


def test_real_shap_additivity_and_aggregation(calibration):
    pytest.importorskip('shap')
    d=weather(24);d.loc[0,'temperature']=55.
    r=run_lstm_autoencoder_detector(d,calibration).iloc[0].to_dict();r['final_alert']=True
    background=weather(96);background.timestamp-=pd.Timedelta(days=365)
    context=ReconstructionShap(d,calibration,background,ShapConfig(background_size=2,max_rows=1,permutations=1,max_seconds=60))
    result=context.explain(r)
    assert result['available'],result
    assert result['base_value']+sum(x['contribution'] for x in result['timestep_contributions'])==pytest.approx(r['lstm_ae_score'],rel=1e-6)
    for v,magnitude in result['variable_contributions'].items():
        assert magnitude==pytest.approx(sum(abs(x['contribution']) for x in result['timestep_contributions'] if x['variable']==v))
    assert result['background_size']==2
    assert context.explain(r) is result
    assert len(result['timestep_contributions'])==72


def test_prediction_budget_fallback(calibration):
    d=weather();r=run_lstm_autoencoder_detector(d,calibration).iloc[24].to_dict();r['final_alert']=True
    bg=weather(96);bg.timestamp-=pd.Timedelta(days=365)
    output=ReconstructionShap(d,calibration,bg,ShapConfig(max_model_windows=1)).explain(r)
    assert not output['available'] and 'budget exceeded' in output['reason']


def test_spatial_humidity_not_claimed_as_vote_driver():
    d=row();d.update(humidity_spatial_anomaly=True,humidity_spatial_residual=30.)
    assert detector_evidence(d)['spatial']['primary_variable'] is None


def test_window_preparation_drops_synthetic_labels(calibration):
    d=weather();clean=normalized_windows(d,calibration)
    poison=d.assign(is_anomaly=True,anomaly_type='spike',anomaly_variable='humidity',event_id='fake',temperature_original=9000.)
    prepared=normalized_windows(poison,calibration)
    pd.testing.assert_frame_equal(clean[0],prepared[0],check_exact=True)
    pd.testing.assert_frame_equal(clean[1],prepared[1],check_exact=True)
    np.testing.assert_array_equal(clean[2],prepared[2])
    np.testing.assert_array_equal(clean[3],prepared[3])


def test_diagnosis_adapter_with_existing_reference():
    from skyguard.diagnostics.diagnosis import DiagnosisReference
    d=weather(6).assign(final_alert=True)
    for v in ['temperature','pressure','humidity']:
        d[v]=20.+np.arange(6)*.2+(6. if v=='temperature' else 0.)
        d[f'{v}_spatial_expected']=20.+np.arange(6)*.2
        d[f'{v}_spatial_neighbor_count']=3
    reference=DiagnosisReference(pd.DataFrame([dict(station_id='AWS_001',variable=v,
        residual_center=0.,residual_scale=1.,rate_center=0.,rate_scale=1.,rate_threshold=3.)
        for v in ['temperature','pressure','humidity']]),'2025-12-31')
    a=diagnose_fused(d,reference)
    assert a.diagnosis.iloc[-1]=='SENSOR_OFFSET'
    b=diagnose_fused(d.assign(is_anomaly=False,anomaly_type='noise',event_id='fake'),reference)
    assert a.diagnosis.tolist()==b.diagnosis.tolist()


def test_row_time_budget_fallback(calibration):
    d=weather(24);r=run_lstm_autoencoder_detector(d,calibration).iloc[0].to_dict();r['final_alert']=True
    bg=weather(96);bg.timestamp-=pd.Timedelta(days=365)
    output=ReconstructionShap(d,calibration,bg,ShapConfig(max_seconds=1e-12)).explain(r)
    assert not output['available'] and 'time budget exceeded' in output['reason']


def test_nonfinite_evidence_is_json_safe():
    import json
    d=row();d.update(temperature=np.inf,temperature_lstm_ae_error=np.inf,
                    temperature_lstm_ae_expected=20.,temperature_lstm_ae_residual=np.inf)
    json.dumps(explain_alert(d),allow_nan=False)
