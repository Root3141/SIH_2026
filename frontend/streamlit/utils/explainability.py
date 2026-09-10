"""Small optional operator panel; reads precomputed evidence without inference."""
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0,str(ROOT/'src'))
from skyguard.explainability import explain_alert


@st.cache_data(max_entries=32)
def _load_detail(path, modified, station, timestamp):
    data=pd.read_parquet(path,filters=[('station_id','=',station),('timestamp','=',timestamp)])
    return json.loads(data.explanation.iloc[0]) if len(data)==1 else None


def render_explanation(current):
    explanation=explain_alert(current)
    path=ROOT/'results/explainability/operator_explanations.parquet'
    if path.exists():
        try:
            candidate=_load_detail(str(path),path.stat().st_mtime_ns,current['station_id'],current['timestamp'])
            # Refuse stale decisions/readings. Full explanations are retrospective
            # batch outputs, just like the detector's overlapping-window scores.
            if candidate and candidate['decision']==explanation['decision']:
                evidence=candidate['detectors']['lstm']['evidence']
                matches=all((evidence[v].get('observed')==float(current[v])) or
                            (evidence[v].get('observed') is None and pd.isna(current[v]))
                            for v in ['temperature','pressure','humidity'])
                if matches:explanation=candidate
        except (OSError,ValueError,KeyError):
            st.caption('Precomputed detail unavailable for this reading.')
    st.markdown('### Why was this alert raised?' if current['final_alert'] else '### Why is there no final alert?')
    for name,vote in explanation['decision']['detector_vote_summary'].items():
        st.write(f"{'✓ anomaly' if vote else '○ no anomaly vote'} — {name.title()}")
    st.write(explanation['decision']['alert_explanation_summary'])
    st.markdown('### What did the detectors observe?')
    for name,detector in explanation['detectors'].items():
        with st.expander('LSTM reconstruction model' if name=='lstm' else name.title()):
            st.write(detector['summary'])
            if detector['primary_variable']:st.write('Primary contributor:',detector['primary_variable'])
            for observation in detector.get('observations',[]):st.write(observation)
            if detector['available']:st.json(detector['evidence'])
    st.markdown('### Model explainability')
    model=explanation['model_explanation']
    if model['available']:
        st.caption('SHAP contribution magnitude to reconstruction anomaly score; not causal importance.')
        st.bar_chart(pd.Series(model['variable_contributions'],name='Model contribution magnitude'))
    else:
        st.caption('Detailed SHAP explanation unavailable; reconstruction-based evidence shown when available.')
    st.markdown('### What may be wrong?')
    diagnosis=explanation['diagnosis']
    if diagnosis['label']:
        st.write('Likely behavior:',diagnosis['label'])
        st.write('Operational class:',diagnosis['operational_class'])
        st.write(diagnosis['reason'])
        st.write(diagnosis['recommendation'])
        if diagnosis['reference_status']:st.caption(diagnosis['reference_status'])
        with st.expander('Diagnosis evidence'):st.json(diagnosis['evidence'])
    else:
        st.caption('Operational diagnosis unavailable or not evaluated for this reading.')
    st.caption('Batch evidence may include later observations. Diagnosis confidence grades are heuristic, not probabilities.')
