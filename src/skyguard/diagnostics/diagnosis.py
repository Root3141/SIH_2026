"""Deterministic Phase-1 diagnosis, using frozen historical behavior normalizers.

Only explicit observation/evidence columns are read. Labels and clean/original
counterfactual readings are never selected. Batch spike confirmation uses one
following hourly reading; all other shape features use six trailing observations.
Confidence is an engineering evidence grade, not a calibrated probability.
"""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd

VARIABLES = ['temperature', 'humidity', 'pressure', 'surface_pressure']
PRECEDENCE = ['DATA_DROPOUT','STUCK_SENSOR','NOISY_SENSOR','SPIKE',
              'SENSOR_DRIFT','SENSOR_OFFSET','RATE_CHANGE','UNCLASSIFIED_ANOMALY']
OUTPUT_COLUMNS = ['diagnosis','diagnosis_variable','diagnosis_confidence',
                  'operational_class','diagnosis_reason','diagnosis_evidence']


@dataclass
class DiagnosisReference:
    parameters: pd.DataFrame
    historical_end: str

    def __post_init__(self):
        self.parameters = self.parameters.copy()
        required = {'station_id','variable','residual_center','residual_scale','rate_center','rate_scale','rate_threshold'}
        if required-set(self.parameters) or self.parameters.duplicated(['station_id','variable']).any():
            raise ValueError('Invalid historical reference schema')
        numeric = self.parameters[list(required-{'station_id','variable'})].to_numpy(float)
        if not np.isfinite(numeric).all() or (self.parameters[['residual_scale','rate_scale','rate_threshold']]<=0).any().any():
            raise ValueError('Invalid historical scales')
        self.historical_end = str(pd.to_datetime(self.historical_end,utc=True))


def _inputs(df, reference):
    required = ['station_id','timestamp','final_alert']+VARIABLES
    if set(required)-set(df) or not df.columns.is_unique:
        raise ValueError('Missing required observation/final-alert columns or duplicate columns')
    if df.final_alert.isna().any() or df.final_alert.dtype != bool:
        raise ValueError('final_alert must be boolean')
    allowed = required + [f'behavior_peer_{kind}_{v}' for v in VARIABLES for kind in ['expected','count']]
    data = df[[c for c in allowed if c in df]].copy().reset_index(drop=True)
    data.timestamp = pd.to_datetime(data.timestamp,utc=True,errors='raise')
    if data[['station_id','timestamp']].isna().any().any() or data.duplicated(['station_id','timestamp']).any():
        raise ValueError('Invalid observation keys')
    if not data.timestamp.eq(data.timestamp.dt.floor('h')).all():
        raise ValueError('Exact hourly timestamps required')
    if len(data) and data.timestamp.min()<=pd.Timestamp(reference.historical_end):
        raise ValueError('Historical reference must strictly precede diagnosis timeline')
    data['_pos'] = np.arange(len(data))
    return data.sort_values(['station_id','timestamp']).reset_index(drop=True)


def _variable_features(data, variable, reference):
    n=len(data)
    names=['z','rate','dx','peer_dx','peer_count','peer_span','span','median','slope','mad',
           'monotonic','same_sign','noise','reversal','spike','ready']
    features=pd.DataFrame(np.nan,index=data.index,columns=names)
    params=reference.parameters.set_index(['station_id','variable'])
    for station,g in data.groupby('station_id',sort=False):
        if (station,variable) not in params.index:
            continue  # No invented normalization for unknown stations.
        p=params.loc[(station,variable)]
        x=g[variable].astype(float).replace([np.inf,-np.inf],np.nan)
        peer=g.get('behavior_peer_expected_'+variable,pd.Series(np.nan,index=g.index)).astype(float)
        count=g.get('behavior_peer_count_'+variable,pd.Series(0,index=g.index))
        peer=peer.where(count.ge(2) & np.isfinite(peer))
        z=(x-peer-p.residual_center)/p.residual_scale
        contiguous=g.timestamp.diff().eq(pd.Timedelta(hours=1))
        dx=x.diff().where(contiguous); dp=peer.diff().where(contiguous)
        dz=z.diff().where(contiguous)
        rate=((x-peer).diff()-p.rate_center)/p.rate_scale
        rate=rate.where(contiguous)
        ready=(z.rolling(6).count().eq(6) & contiguous.rolling(5).sum().eq(5))
        zr=z.rolling(6)
        median=zr.median()
        slope=dz.rolling(5).median()
        mad=zr.apply(lambda a: np.median(np.abs(a-np.median(a))),raw=True)
        same=pd.concat([z.gt(0).rolling(6).mean(),z.lt(0).rolling(6).mean()],axis=1).max(axis=1)
        mono=pd.concat([dz.gt(0).rolling(5).mean(),dz.lt(0).rolling(5).mean()],axis=1).max(axis=1)
        noise=rate.rolling(5).apply(lambda a:np.median(np.abs(a-np.median(a))),raw=True)
        reversal=(dz*dz.shift(1)).lt(0).rolling(4).mean()
        spike=(z.abs().ge(3) & z.shift(1).abs().lt(2) & z.shift(2).abs().lt(2)
               & z.shift(-1).abs().lt(2) & contiguous & contiguous.shift(1,fill_value=False)
               & contiguous.shift(-1,fill_value=False))
        values=dict(z=z,rate=rate,dx=dx,peer_dx=dp,peer_count=count,peer_span=peer.rolling(6).max()-peer.rolling(6).min(),
            span=x.rolling(6).max()-x.rolling(6).min(),median=median,slope=slope,mad=mad,
            monotonic=mono,same_sign=same,noise=noise,reversal=reversal,spike=spike.astype(float),ready=ready.astype(float))
        for name,value in values.items(): features.loc[g.index,name]=value.to_numpy()
    return features


def diagnose(df, reference):
    """Append six fields, preserving all original rows/columns/index exactly.

    Inputs must include unalerted temporal context rows as well as upstream alerts.
    Missing optional peer evidence causes abstention, not imputation or detection.
    """
    if set(OUTPUT_COLUMNS)&set(df):
        raise ValueError('Diagnostic output columns already exist')
    data=_inputs(df,reference)
    features={v:_variable_features(data,v,reference) for v in VARIABLES}
    params=reference.parameters.set_index(['station_id','variable'])
    # Conservative network corroboration, independent of ground truth or votes.
    network={}
    for v,f in features.items():
        normal=f.z.abs().lt(2) & np.isfinite(data[v])
        network[v]=normal.groupby(data.timestamp).transform('sum')-normal.astype(int)
    records=[dict(diagnosis=None,diagnosis_variable=None,diagnosis_confidence=np.nan,
        operational_class='UNCERTAIN',diagnosis_reason='No upstream alert; diagnosis not evaluated.',
        diagnosis_evidence='{}') for _ in range(len(df))]
    for i in np.flatnonzero(data.final_alert):
        candidates=[]
        for v in VARIABLES:
            f=features[v].loc[i]; x=float(data[v].iloc[i]); station=data.station_id.iloc[i]
            kind=None; confidence=.0; reason=''
            if not np.isfinite(x):
                kind='DATA_DROPOUT'; confidence=.95; reason='Original observation is nonfinite; sensor or telemetry availability failure.'
            elif f.ready==1 and f.span<=.05 and f.peer_span>=.2 and f.peer_span>=4*f.span:
                kind='STUCK_SENSOR'; confidence=.85; reason='Six-hour motion collapse while spatial peer reference moves.'
            elif f.ready==1 and f.noise>=3 and f.reversal>=.6:
                kind='NOISY_SENSOR'; confidence=.7; reason='Excess robust residual-difference variability with repeated direction reversals.'
            elif f.spike==1:
                kind='SPIKE'; confidence=.8; reason='Large residual between quiet preceding readings and next-hour recovery; retrospective confirmation.'
            elif f.ready==1 and abs(f['median'])>=3 and f.same_sign>=.8:
                if abs(f.slope)>.25 and f.monotonic>=.8:
                    kind='SENSOR_DRIFT'; confidence=.65; reason='Sustained signed bias with gradual monotonic residual evolution.'
                elif abs(f.slope)<=.25 and f['mad']<=1:
                    kind='SENSOR_OFFSET'; confidence=.65; reason='Sustained signed bias with approximately constant robust residual level.'
            if kind is None and (station,v) in params.index and abs(f.rate)>=params.loc[(station,v),'rate_threshold']:
                kind='RATE_CHANGE'; confidence=.55; reason='Residual derivative exceeds the frozen historical rate threshold; sustained ramp not yet established.'
            if kind:
                strength=abs(f.z) if np.isfinite(f.z) else 0.
                evidence={k:float(val) if np.isfinite(val) else None for k,val in f.items()}
                evidence.update(variable=v,diagnosis=kind,normal_network_peers=int(network[v].iloc[i]))
                candidates.append((PRECEDENCE.index(kind),-strength,VARIABLES.index(v),kind,v,confidence,reason,evidence))
        candidates.sort(key=lambda c:c[:3])
        if candidates:
            _,_,_,kind,v,confidence,reason,evidence=candidates[0]
            f=features[v].loc[i]
            operational='SENSOR_FAULT' if np.isfinite(f.z) and abs(f.z)>=3 and f.peer_count>=2 and network[v].iloc[i]>=2 else 'UNCERTAIN'
        else:
            kind='UNCLASSIFIED_ANOMALY';v=None;confidence=0.;reason='Available shape evidence does not justify a specific failure mode.';operational='UNCERTAIN'
        # Correlated movement is only suggestive, never a confirmed weather event.
        operational_support={}
        if operational=='UNCERTAIN' and kind!='DATA_DROPOUT':
            for variable,fset in features.items():
                f=fset.loc[i]
                if not (np.isfinite(f.dx) and abs(f.dx)>.05 and abs(f.z)<2 and f.peer_count>=2): continue
                if (data.station_id.iloc[i],variable) not in params.index: continue
                if abs(f.dx)/params.loc[(data.station_id.iloc[i],variable),'rate_scale']<3: continue
                same_time=data.timestamp.eq(data.timestamp.iloc[i]) & data.station_id.ne(data.station_id.iloc[i])
                peers=fset.loc[same_time]
                agree=((peers.dx*f.dx)>0) & peers.dx.abs().ge(.75*abs(f.dx)) & peers.dx.abs().le(1.25*abs(f.dx))
                if agree.sum()>=2 and abs(f.peer_dx-f.dx)<=.25*abs(f.dx):
                    operational='POSSIBLE_ENVIRONMENTAL_EVENT'
                    operational_support=dict(variable=variable,agreeing_network_stations=int(agree.sum()),
                        observed_difference=float(f.dx),peer_difference=float(f.peer_dx))
                    reason+=' At least two network stations and the spatial reference move similarly.'
                    break
        records[int(data._pos.iloc[i])]=dict(diagnosis=kind,diagnosis_variable=v,
            diagnosis_confidence=confidence,operational_class=operational,diagnosis_reason=reason,
            diagnosis_evidence=json.dumps({'candidates':[c[-1] for c in candidates],
                'operational_support':operational_support,'precedence':PRECEDENCE,
                'confidence_is_probability':False,'spike_confirmation_lookahead_hours':1},sort_keys=True,allow_nan=False))
    result=df.copy()
    for c in OUTPUT_COLUMNS: result[c]=[r[c] for r in records]
    return result
