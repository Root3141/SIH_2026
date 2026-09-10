"""Recover omitted evidence using unchanged frozen detectors, outside fusion."""
from pathlib import Path
import pandas as pd
from skyguard.fusion.fusion import ARTIFACTS_DIR, DETECTOR_COLUMNS, KEY_COLUMNS, VARIABLES, _load_artifacts
from skyguard.detectors.statistical import run_statistical_detector
from skyguard.detectors.lstm_autoencoder import run_lstm_autoencoder_detector


def enrich_evidence(fused, artifacts_dir=ARTIFACTS_DIR):
    if fused.duplicated(KEY_COLUMNS).any():
        raise ValueError('Duplicate station/timestamp keys')
    weather = fused[DETECTOR_COLUMNS].copy().sort_values(KEY_COLUMNS).reset_index(drop=True)
    stat, _, lstm = _load_artifacts(Path(artifacts_dir))
    outputs = [run_statistical_detector(weather, VARIABLES, stat['calibration'], stat['config']),
               run_lstm_autoencoder_detector(weather, lstm, lstm.config)]
    result = fused.copy()
    keys = pd.MultiIndex.from_frame(fused[KEY_COLUMNS])
    for output in outputs:
        aligned = output.set_index(KEY_COLUMNS).loc[keys].reset_index()
        for column in output:
            if column in KEY_COLUMNS:
                continue
            if column in fused:
                pd.testing.assert_series_equal(fused[column].reset_index(drop=True), aligned[column], check_exact=True)
            else:
                result[column] = aligned[column].to_numpy()
    return result
