"""Build the historical DiagnosisReference CSV from clean 2023-2025 data.

Uses the frozen spatial detector so residual_center/residual_scale are computed
against the exact same peer-expectation definition diagnosis_adapter.py feeds
at runtime (`{v}_spatial_expected`, gated on `{v}_spatial_neighbor_count`>=2).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from skyguard.config.paths import PROJECT_ROOT, HISTORICAL_PARQUET
from skyguard.fusion.fusion import ARTIFACTS_DIR, _load_artifacts
from skyguard.detectors.spatial import run_spatial_detector

VARIABLES = ['temperature', 'humidity', 'pressure']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', type=Path, default=HISTORICAL_PARQUET)
    parser.add_argument('--historical-end', required=True)
    parser.add_argument('--artifacts-dir', type=Path, default=ARTIFACTS_DIR)
    parser.add_argument('--rate-z-threshold', type=float, default=3.0)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/reference/diagnosis_reference.csv')
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.historical_end, tz='UTC')
    df = pd.read_parquet(args.historical)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df[df.timestamp <= cutoff].sort_values(['station_id', 'timestamp']).reset_index(drop=True)

    stat, spatial, lstm = _load_artifacts(args.artifacts_dir)
    scored = run_spatial_detector(df, VARIABLES, spatial['calibration'], spatial['config'])

    rows = []
    for station, g in scored.groupby('station_id'):
        g = g.sort_values('timestamp')
        contiguous = g.timestamp.diff().eq(pd.Timedelta(hours=1))
        for v in VARIABLES:
            expected = g[f'{v}_spatial_expected']
            count = g[f'{v}_spatial_neighbor_count']
            peer = expected.where(count.ge(2) & np.isfinite(expected))
            residual = (g[v].astype(float) - peer).replace([np.inf, -np.inf], np.nan)
            if residual.notna().sum() < 30:
                continue
            residual_center = float(residual.mean())
            residual_scale = float(residual.std())
            if not np.isfinite(residual_scale) or residual_scale <= 0:
                continue
            rate = residual.diff().where(contiguous).dropna()
            if len(rate) < 30:
                continue
            rate_center = float(rate.mean())
            rate_scale = float(rate.std())
            if not np.isfinite(rate_scale) or rate_scale <= 0:
                continue
            rows.append(dict(station_id=station, variable=v,
                residual_center=residual_center, residual_scale=residual_scale,
                rate_center=rate_center, rate_scale=rate_scale,
                rate_threshold=args.rate_z_threshold))

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'wrote {len(out)} station/variable rows to {args.output}')


if __name__ == '__main__':
    main()