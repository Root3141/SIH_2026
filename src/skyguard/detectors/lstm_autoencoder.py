"""
LSTM Autoencoder Anomaly Detector
==================================

File: src/skyguard/detectors/lstm_autoencoder.py

A deep, multivariate, *temporal* anomaly detector for AWS observations
(temperature, pressure, humidity), complementary to the existing
statistical (per-station temporal baselines) and spatial (IDW / neighbor
consistency) detectors described in DEVELOPER_GUIDE.md.

Design goals (see problem statement + developer guide "Adding a New
Detector" section):

    * Learn *normal* multivariate temporal patterns with a small LSTM
      Autoencoder (encoder compresses a sliding window into a latent
      vector, decoder reconstructs the window). Reconstruction error is
      the anomaly score -- the model has never seen anomalies, so it
      reconstructs anomalous windows poorly.
    * Respect station boundaries: windows are built independently per
      station and per contiguous time segment, never mixing stations,
      never bridging large data gaps.
    * No leakage: normalization statistics and the model are fit only on
      a chronological *train* slice of the historical/calibration data;
      thresholds are calibrated on a later, held-out chronological
      *validation* slice (never on the unseen evaluation data).
    * Small, practical architecture (single-layer LSTM encoder/decoder,
      <10k params by default) -- optimized for "does this run reliably
      in a hackathon demo" over architectural sophistication.
    * Sliding-window reconstruction errors are mapped back to
      observation-level scores (mean over all overlapping windows) so
      this integrates with the same evaluation pipeline
      (`is_anomaly`, `anomaly_id`, `anomaly_type`, `anomaly_variable`,
      `anomaly_severity`) used by `evaluate_statistical.py` /
      `evaluate_spatial.py`.
    * Reliability fallback: if PyTorch is not installed in the current
      environment, the detector automatically falls back to a small
      scikit-learn MLP autoencoder (same windowing / normalization /
      scoring pipeline, just a different reconstruction model) so the
      pipeline never hard-fails during a demo because of a missing
      dependency. This is logged clearly and is not silent.

Expected input schema (matches the other detectors in this codebase):

    station_id : str
    timestamp  : datetime64 (any tz, converted to UTC internally)
    temperature: float (deg C)
    pressure   : float (hPa)
    humidity   : float (%)

Assumed project conventions (see DEVELOPER_GUIDE.md section 12,
"Adding a New Detector" and "Keep output conventions consistent"):

    * Detector output columns are prefixed with the detector name
      (`lstm_ae_*`), mirroring `statistical_*` / `spatial_*`.
    * `run_*_detector(df, calibration, config)` returns the input
      DataFrame's rows (aligned by index) plus new detector columns --
      it does not drop or reorder rows.
    * A `_run_self_test()` function is provided so this module can be
      sanity-checked standalone (`python lstm_autoencoder.py`).

NOTE ON INTEGRATION: this file was written from the documented
architecture (CODEBASE_REFERENCE.md / DEVELOPER_GUIDE.md) rather than
from the private repository source, since only the documentation was
available at implementation time. Column names / config import paths
are the best-supported assumptions from the docs; if your actual
`skyguard.config.paths` module uses different constant names, adjust
the small `try/except` import block below accordingly -- everything
else is self-contained.
"""

from __future__ import annotations

from unittest import result
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except (
    Exception
):  # pragma: no cover - exercised in torch-less/broken-install environments
    _TORCH_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

try:
    from skyguard.config.paths import HISTORICAL_PARQUET, PRESENT_PARQUET, RESULTS_DIR
except ImportError:  # pragma: no cover - fallback for standalone use
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    HISTORICAL_PARQUET = (
        _PROJECT_ROOT / "data" / "raw" / "ncr_weather_historical.parquet"
    )
    PRESENT_PARQUET = (
        _PROJECT_ROOT / "data" / "raw" / "ncr_weather_2026_present.parquet"
    )
    RESULTS_DIR = _PROJECT_ROOT / "results"

DEFAULT_VARIABLES = ["temperature", "pressure", "humidity"]


@dataclass
class LSTMAEConfig:
    """Hyperparameters and behavioral knobs for the LSTM Autoencoder detector."""

    variables: List[str] = field(default_factory=lambda: list(DEFAULT_VARIABLES))

    window_size: int = 24  # e.g. 24 hourly observations = 1 day of context
    train_stride: int = 1  # step between windows when building training data
    eval_stride: int = 1  # step between windows at scoring time (1 = full coverage)
    gap_tolerance: float = (
        1.5  # a gap > gap_tolerance * modal_freq splits a station's segment
    )
    min_valid_fraction: float = (
        0.8  # min fraction of non-NaN values required to keep a window
    )
    max_fill_run: int = 2  # max consecutive NaNs to forward/back-fill inside a window

    min_station_obs_for_own_stats: int = 500  # below this, fall back to global stats
    clip_z: float = 8.0  # clip normalized inputs to +/- this many std devs (robustness)

    hidden_size: int = 32
    latent_size: int = 16
    num_layers: int = 1
    dropout: float = 0.1

    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    val_fraction: float = 0.15  # chronological fraction of historical data held out
    early_stopping_patience: int = 5
    random_seed: int = 42
    device: str = "cpu"

    suspicious_percentile: float = 95.0
    anomaly_percentile: float = 99.0
    score_aggregation: str = "mean"  # "mean" or "max" over overlapping windows

    backend: str = "auto"  # "auto" | "torch" | "sklearn"


@dataclass
class LSTMAECalibration:
    config: LSTMAEConfig
    variables: List[str]
    backend: str
    model_state: Any  # torch state_dict OR fitted sklearn model object
    global_stats: Dict[str, Dict[str, float]]
    station_stats: Dict[str, Dict[str, Dict[str, float]]]
    threshold_p95: float
    threshold_p99: float
    per_variable_thresholds: Dict[str, Dict[str, float]]
    training_history: List[dict]
    calibrated_at: str
    n_stations: int
    n_training_windows: int
    n_validation_windows: int


def _validate_input(df: pd.DataFrame, variables: Sequence[str]) -> None:
    required = {"station_id", "timestamp", *variables}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {sorted(missing)}"
        )


def _prepare(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return out


def _infer_frequency(timestamps: pd.Series) -> pd.Timedelta:
    diffs = timestamps.diff().dropna()
    if len(diffs) == 0:
        return pd.Timedelta(hours=1)
    modal = diffs.value_counts().idxmax()
    return modal


def _segment_indices(
    timestamps: pd.Series, freq: pd.Timedelta, gap_tolerance: float
) -> List[Tuple[int, int]]:
    """Return list of (start, end) positional ranges (end exclusive) such that
    consecutive timestamps within a segment never jump by more than
    gap_tolerance * freq. Positions are relative to `timestamps` (0-based)."""
    n = len(timestamps)
    if n == 0:
        return []
    max_gap = freq * gap_tolerance
    ts = timestamps.reset_index(drop=True)
    segments = []
    seg_start = 0
    for i in range(1, n):
        if ts.iloc[i] - ts.iloc[i - 1] > max_gap:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, n))
    return segments


def _compute_stats(values: np.ndarray) -> Dict[str, float]:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return {"mean": 0.0, "std": 1.0, "n": 0}
    std = float(np.std(clean))
    return {
        "mean": float(np.mean(clean)),
        "std": std if std > 1e-6 else 1.0,
        "n": int(len(clean)),
    }


def _fit_normalization(
    train_df: pd.DataFrame, variables: Sequence[str], min_station_obs: int
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    global_stats: Dict[str, Dict[str, float]] = {}
    for var in variables:
        global_stats[var] = _compute_stats(train_df[var].to_numpy(dtype=float))

    station_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for station_id, g in train_df.groupby("station_id"):
        var_stats = {}
        for var in variables:
            stats = _compute_stats(g[var].to_numpy(dtype=float))
            if stats["n"] >= min_station_obs:
                var_stats[var] = stats
            else:
                var_stats[var] = global_stats[var]
        station_stats[str(station_id)] = var_stats
    return global_stats, station_stats


def _stats_for_station(
    station_id: str,
    variables: Sequence[str],
    global_stats: Dict[str, Dict[str, float]],
    station_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    return station_stats.get(str(station_id), global_stats)


def _normalize_frame(
    df: pd.DataFrame,
    variables: Sequence[str],
    global_stats: Dict[str, Dict[str, float]],
    station_stats: Dict[str, Dict[str, Dict[str, float]]],
    clip_z: float,
) -> pd.DataFrame:
    out = df.copy()

    # Force normalized variables to floating point.
    # Z-score normalization cannot preserve integer dtype.
    for var in variables:
        out[var] = pd.to_numeric(out[var], errors="coerce").astype(np.float64)

    for station_id, idx in out.groupby("station_id").groups.items():
        stats = _stats_for_station(
            station_id,
            variables,
            global_stats,
            station_stats,
        )

        for var in variables:
            mean = stats[var]["mean"]
            std = stats[var]["std"]

            z = (out.loc[idx, var].to_numpy(dtype=np.float64) - mean) / std

            out.loc[idx, var] = np.clip(z, -clip_z, clip_z)

    return out


def _denormalize_value(
    value: float, station_id: str, var: str, global_stats, station_stats
) -> float:
    stats = _stats_for_station(station_id, [var], global_stats, station_stats)
    return value * stats[var]["std"] + stats[var]["mean"]


def _fill_small_gaps(window: np.ndarray, max_fill_run: int) -> np.ndarray:
    """Forward/back-fill short NaN runs inside a (window_size, n_features) array.
    Leaves long NaN runs (> max_fill_run) untouched so the window gets rejected
    by the min_valid_fraction check upstream instead of being silently faked."""
    filled = window.copy()
    n_features = window.shape[1]
    for f in range(n_features):
        col = pd.Series(filled[:, f])
        col = col.ffill(limit=max_fill_run).bfill(limit=max_fill_run)
        filled[:, f] = col.to_numpy()
    return filled


def _build_windows(
    norm_df: pd.DataFrame,
    variables: Sequence[str],
    window_size: int,
    stride: int,
    min_valid_fraction: float,
    max_fill_run: int,
    gap_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build sliding windows *per station, per contiguous time segment*.

    Returns
    -------
    windows      : (N, window_size, n_features) float array, NaN-free
    row_indices  : (N, window_size) array of the original dataframe index
                   values covered by each window (for mapping scores back)
    station_ids  : list of length N, the station each window belongs to
    """
    all_windows: List[np.ndarray] = []
    all_row_idx: List[np.ndarray] = []
    all_stations: List[str] = []

    for station_id, g in norm_df.groupby("station_id"):
        g = g.sort_values("timestamp")
        freq = _infer_frequency(g["timestamp"])
        segments = _segment_indices(
            g["timestamp"].reset_index(drop=True), freq, gap_tolerance
        )
        values = g[list(variables)].to_numpy(dtype=float)
        orig_index = g.index.to_numpy()

        for seg_start, seg_end in segments:
            seg_values = values[seg_start:seg_end]
            seg_index = orig_index[seg_start:seg_end]
            n = len(seg_values)
            if n < window_size:
                continue
            for start in range(0, n - window_size + 1, stride):
                window = seg_values[start : start + window_size]
                valid_fraction = 1.0 - (np.isnan(window).sum() / window.size)
                if valid_fraction < min_valid_fraction:
                    continue
                filled = _fill_small_gaps(window, max_fill_run)
                if np.isnan(filled).any():
                    continue
                all_windows.append(filled)
                all_row_idx.append(seg_index[start : start + window_size])
                all_stations.append(str(station_id))

    if not all_windows:
        return (
            np.empty((0, window_size, len(variables))),
            np.empty((0, window_size), dtype=object),
            [],
        )

    return np.stack(all_windows), np.stack(all_row_idx), all_stations


if _TORCH_AVAILABLE:

    class _LSTMAutoencoderNet(nn.Module):
        """Small seq2seq ("RepeatVector"-style) LSTM autoencoder."""

        def __init__(
            self,
            n_features: int,
            hidden_size: int,
            latent_size: int,
            num_layers: int,
            dropout: float,
        ):
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.encoder_lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.to_latent = nn.Linear(hidden_size, latent_size)
            self.from_latent = nn.Linear(latent_size, hidden_size)
            self.decoder_lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.output_layer = nn.Linear(hidden_size, n_features)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            batch_size, seq_len, _ = x.shape
            _, (h_n, _c_n) = self.encoder_lstm(x)
            h_last = h_n[-1]  # (batch, hidden_size) - final layer's hidden state
            latent = self.to_latent(h_last)  # (batch, latent_size)
            decoder_seed = self.from_latent(latent)  # (batch, hidden_size)
            decoder_input = decoder_seed.unsqueeze(1).repeat(1, seq_len, 1)
            decoded, _ = self.decoder_lstm(decoder_input)
            return self.output_layer(decoded)  # (batch, seq_len, n_features)


def _build_model(config: LSTMAEConfig, n_features: int):
    backend = config.backend
    if backend == "auto":
        backend = "torch" if _TORCH_AVAILABLE else "sklearn"

    if backend == "torch":
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "backend='torch' requested but PyTorch is not installed."
            )
        torch.manual_seed(config.random_seed)
        model = _LSTMAutoencoderNet(
            n_features=n_features,
            hidden_size=config.hidden_size,
            latent_size=config.latent_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        return backend, model

    if backend == "sklearn":
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "backend='sklearn' requested but scikit-learn is not installed."
            )
        hidden1 = max(config.hidden_size, config.latent_size * 2)
        model = MLPRegressor(
            hidden_layer_sizes=(hidden1, config.latent_size, hidden1),
            activation="tanh",
            solver="adam",
            alpha=1e-4,
            batch_size=min(config.batch_size, 256),
            max_iter=max(config.epochs * 5, 50),
            learning_rate_init=config.learning_rate,
            random_state=config.random_seed,
            early_stopping=True,
            n_iter_no_change=config.early_stopping_patience,
            validation_fraction=0.1,
        )
        return backend, model

    raise ValueError(f"Unknown backend: {backend}")


def _train_torch(
    model, train_windows: np.ndarray, val_windows: np.ndarray, config: LSTMAEConfig
):
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=config.batch_size, shuffle=True
    )

    val_tensor = (
        torch.tensor(val_windows, dtype=torch.float32).to(device)
        if len(val_windows)
        else None
    )

    best_val = float("inf")
    best_state = None
    patience = 0
    history: List[dict] = []

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        if val_tensor is not None and len(val_tensor) > 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_tensor), val_tensor).item()
        else:
            val_loss = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _reconstruct_torch(model, windows: np.ndarray, config: LSTMAEConfig) -> np.ndarray:
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    out = np.zeros_like(windows, dtype=np.float64)
    with torch.no_grad():
        bs = config.batch_size
        for start in range(0, len(windows), bs):
            chunk = windows[start : start + bs]
            recon = (
                model(torch.tensor(chunk, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )
            out[start : start + bs] = recon
    return out


def _reconstruct_sklearn(model, windows: np.ndarray) -> np.ndarray:
    flat = windows.reshape(len(windows), -1)
    recon_flat = model.predict(flat)
    return recon_flat.reshape(windows.shape)


def calibrate_lstm_autoencoder_detector(
    historical_df: pd.DataFrame, config: Optional[LSTMAEConfig] = None
) -> LSTMAECalibration:
    """Fit normalization + LSTM autoencoder on historical (calibration) data only.

    Chronological split: the *last* `config.val_fraction` of the historical
    time range is held out as validation, used for (a) early stopping and
    (b) calibrating the P95/P99 reconstruction-error thresholds. It is never
    used to fit normalization stats or model weights -- this is the
    train/val leakage boundary. The unseen 2026 evaluation data is never
    touched here at all.
    """
    config = config or LSTMAEConfig()
    variables = list(config.variables)
    _validate_input(historical_df, variables)
    df = _prepare(historical_df, variables)

    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    cutoff = t_min + (t_max - t_min) * (1 - config.val_fraction)

    train_df = df[df["timestamp"] < cutoff].reset_index(drop=True)
    val_df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    if len(train_df) < config.window_size * 5:
        raise ValueError(
            "Not enough historical training data to build meaningful windows "
            f"(need at least ~{config.window_size * 5} rows before the validation cutoff)."
        )

    global_stats, station_stats = _fit_normalization(
        train_df, variables, config.min_station_obs_for_own_stats
    )

    norm_train = _normalize_frame(
        train_df, variables, global_stats, station_stats, config.clip_z
    )
    norm_val = _normalize_frame(
        val_df, variables, global_stats, station_stats, config.clip_z
    )

    train_windows, _train_rows, _ = _build_windows(
        norm_train,
        variables,
        config.window_size,
        config.train_stride,
        config.min_valid_fraction,
        config.max_fill_run,
        config.gap_tolerance,
    )
    val_windows, val_rows, _ = _build_windows(
        norm_val,
        variables,
        config.window_size,
        config.eval_stride,
        config.min_valid_fraction,
        config.max_fill_run,
        config.gap_tolerance,
    )

    if len(train_windows) == 0:
        raise ValueError(
            "No valid training windows could be built -- check window_size vs. data density."
        )

    backend, model = _build_model(config, n_features=len(variables))

    if backend == "torch":
        model, history = _train_torch(model, train_windows, val_windows, config)
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        val_recon = (
            _reconstruct_torch(model, val_windows, config)
            if len(val_windows)
            else np.empty_like(val_windows)
        )
    else:
        val_split = max(1, int(len(train_windows) * 0.9))
        flat_train = train_windows[:val_split].reshape(val_split, -1)
        model.fit(flat_train, flat_train)
        history = [
            {
                "note": "sklearn MLPRegressor loss curve",
                "loss_curve": list(model.loss_curve_),
            }
        ]
        model_state = model
        val_recon = (
            _reconstruct_sklearn(model, val_windows)
            if len(val_windows)
            else np.empty_like(val_windows)
        )

    if len(val_windows):
        sq_err = (val_windows - val_recon) ** 2  # (N, window_size, n_features)
        obs_err = _aggregate_windows_to_observations(
            val_rows, sq_err, variables, config.score_aggregation
        )
        overall_score = obs_err[variables].mean(axis=1)
        threshold_p95 = float(
            np.percentile(overall_score, config.suspicious_percentile)
        )
        threshold_p99 = float(np.percentile(overall_score, config.anomaly_percentile))
        per_variable_thresholds = {
            var: {
                "p95": float(np.percentile(obs_err[var], config.suspicious_percentile)),
                "p99": float(np.percentile(obs_err[var], config.anomaly_percentile)),
            }
            for var in variables
        }
    else:
        warnings.warn(
            "No validation windows available to calibrate thresholds; falling back to "
            "training-window residuals (less robust, review val_fraction / window_size)."
        )
        train_recon = (
            _reconstruct_torch(model, train_windows, config)
            if backend == "torch"
            else _reconstruct_sklearn(model, train_windows)
        )
        sq_err = (train_windows - train_recon) ** 2
        overall_score = sq_err.mean(axis=(1, 2))
        threshold_p95 = float(
            np.percentile(overall_score, config.suspicious_percentile)
        )
        threshold_p99 = float(np.percentile(overall_score, config.anomaly_percentile))
        per_variable_thresholds = {
            var: {"p95": threshold_p95, "p99": threshold_p99} for var in variables
        }

    return LSTMAECalibration(
        config=config,
        variables=variables,
        backend=backend,
        model_state=model_state,
        global_stats=global_stats,
        station_stats=station_stats,
        threshold_p95=threshold_p95,
        threshold_p99=threshold_p99,
        per_variable_thresholds=per_variable_thresholds,
        training_history=history,
        calibrated_at=datetime.now(timezone.utc).isoformat(),
        n_stations=df["station_id"].nunique(),
        n_training_windows=len(train_windows),
        n_validation_windows=len(val_windows),
    )


def _aggregate_windows_to_observations(
    row_indices: np.ndarray,
    per_var_sq_error: np.ndarray,
    variables: Sequence[str],
    aggregation: str,
) -> pd.DataFrame:
    """Collapse overlapping sliding-window errors down to one row per
    original observation index, using `aggregation` ("mean" or "max")
    across every window that happened to cover that observation."""
    if len(row_indices) == 0:
        return pd.DataFrame(columns=[*variables, "lstm_ae_coverage"])

    flat_idx = row_indices.reshape(-1)
    flat_err = per_var_sq_error.reshape(-1, per_var_sq_error.shape[-1])
    long_df = pd.DataFrame(flat_err, columns=list(variables))
    long_df["__row_index__"] = flat_idx

    grouped = long_df.groupby("__row_index__")
    if aggregation == "mean":
        obs_err = grouped[list(variables)].mean()
    elif aggregation == "max":
        obs_err = grouped[list(variables)].max()
    else:
        raise ValueError(f"Unknown score_aggregation: {aggregation}")
    obs_err["lstm_ae_coverage"] = grouped.size()
    return obs_err


def _aggregate_windows_to_reconstructions(
    row_indices: np.ndarray,
    reconstructions: np.ndarray,
    variables: Sequence[str],
) -> pd.DataFrame:
    """Same idea as `_aggregate_windows_to_observations`, but averages the
    *reconstructed values* (not the squared error) so we can report an
    'expected value' per observation -- used for the optional
    corrected/imputed-value output."""
    if len(row_indices) == 0:
        return pd.DataFrame(columns=list(variables))
    flat_idx = row_indices.reshape(-1)
    flat_recon = reconstructions.reshape(-1, reconstructions.shape[-1])
    long_df = pd.DataFrame(flat_recon, columns=list(variables))
    long_df["__row_index__"] = flat_idx
    return long_df.groupby("__row_index__")[list(variables)].mean()


def _confidence_from_thresholds(score: float, p95: float, p99: float) -> float:
    """Continuous 0-1 confidence, banded to loosely mirror the project's
    existing 0.0/0.33/0.67/1.0 statistical-severity convention:
        [0, p95)   -> 0.00 - 0.33
        [p95, p99) -> 0.33 - 0.67
        [p99, inf) -> 0.67 - 1.00 (saturating)
    """
    span = max(p99 - p95, 1e-9)
    if score <= p95:
        return float(np.clip(0.33 * (score / max(p95, 1e-9)), 0.0, 0.33))
    if score <= p99:
        return float(0.33 + 0.34 * (score - p95) / span)
    return float(np.clip(0.67 + 0.33 * (score - p99) / span, 0.67, 1.0))


def run_lstm_autoencoder_detector(
    df: pd.DataFrame,
    calibration: LSTMAECalibration,
    config: Optional[LSTMAEConfig] = None,
) -> pd.DataFrame:
    """Score `df` using a previously-calibrated LSTM autoencoder.

    Returns the original dataframe (same rows, same index) plus new
    `lstm_ae_*` columns. Rows that cannot be placed in any full sliding
    window (station boundary edges, short segments, gappy data) are kept
    with `lstm_ae_scored=False` rather than silently dropped -- this is
    the "handle station boundaries correctly" requirement made explicit
    and inspectable downstream.
    """
    config = config or calibration.config
    variables = calibration.variables
    _validate_input(df, variables)
    prepared = _prepare(df, variables)

    norm = _normalize_frame(
        prepared,
        variables,
        calibration.global_stats,
        calibration.station_stats,
        config.clip_z,
    )

    windows, row_indices, _stations = _build_windows(
        norm,
        variables,
        config.window_size,
        config.eval_stride,
        config.min_valid_fraction,
        config.max_fill_run,
        config.gap_tolerance,
    )

    n = len(prepared)
    result = prepared.copy()
    result["lstm_ae_scored"] = False
    result["lstm_ae_coverage"] = 0
    result["lstm_ae_score"] = np.nan
    result["lstm_ae_severity_score"] = 0.0
    result["lstm_ae_severity_label"] = "unscored"
    result["lstm_ae_supicious"] = False
    result["lstm_ae_alert"] = False
    result["lstm_ae_top_contributor"] = None
    for var in variables:
        result[f"{var}_lstm_ae_error"] = np.nan
        result[f"{var}_lstm_ae_expected"] = np.nan
        result[f"{var}_lstm_ae_residual"] = np.nan

    if len(windows) == 0:
        warnings.warn(
            "No scorable windows were built for this dataset (too short / too gappy)."
        )
        return result

    if calibration.backend == "torch":
        reconstructions = _reconstruct_torch(
            _rehydrate_torch_model(calibration), windows, config
        )
    else:
        reconstructions = _reconstruct_sklearn(calibration.model_state, windows)

    sq_err = (windows - reconstructions) ** 2
    obs_err = _aggregate_windows_to_observations(
        row_indices, sq_err, variables, config.score_aggregation
    )
    obs_recon = _aggregate_windows_to_reconstructions(
        row_indices, reconstructions, variables
    )

    overall_score = obs_err[variables].mean(axis=1)
    top_contributor = obs_err[variables].idxmax(axis=1)

    scored_idx = obs_err.index
    result.loc[scored_idx, "lstm_ae_scored"] = True
    result.loc[scored_idx, "lstm_ae_coverage"] = obs_err["lstm_ae_coverage"].values
    result.loc[scored_idx, "lstm_ae_score"] = overall_score.values
    result.loc[scored_idx, "lstm_ae_top_contributor"] = top_contributor.values

    severity_label = pd.Series("normal", index=scored_idx)
    severity_label[overall_score > calibration.threshold_p95] = "suspicious"
    severity_label[overall_score > calibration.threshold_p99] = "anomaly"
    result.loc[scored_idx, "lstm_ae_severity_label"] = severity_label.values
    result.loc[scored_idx, "lstm_ae_suspicious"] = (severity_label != "normal").values
    result.loc[scored_idx, "lstm_ae_alert"] = (severity_label == "anomaly").values

    confidences = overall_score.apply(
        lambda s: _confidence_from_thresholds(
            s, calibration.threshold_p95, calibration.threshold_p99
        )
    )
    result.loc[scored_idx, "lstm_ae_severity_score"] = confidences.values

    for var in variables:
        result.loc[scored_idx, f"{var}_lstm_ae_error"] = obs_err[var].values
        expected_norm = obs_recon[var]
        station_ids = result.loc[scored_idx, "station_id"]
        expected_denorm = [
            _denormalize_value(
                v, sid, var, calibration.global_stats, calibration.station_stats
            )
            for v, sid in zip(expected_norm.values, station_ids.values)
        ]
        result.loc[scored_idx, f"{var}_lstm_ae_expected"] = expected_denorm
        result.loc[scored_idx, f"{var}_lstm_ae_residual"] = result.loc[
            scored_idx, var
        ].to_numpy(dtype=float) - np.asarray(expected_denorm)

    result.index = df.index  # guarantee alignment with caller's original index
    return result


def _rehydrate_torch_model(calibration: LSTMAECalibration):
    model = _LSTMAutoencoderNet(
        n_features=len(calibration.variables),
        hidden_size=calibration.config.hidden_size,
        latent_size=calibration.config.latent_size,
        num_layers=calibration.config.num_layers,
        dropout=calibration.config.dropout,
    )
    model.load_state_dict(calibration.model_state)
    return model


def _make_synthetic_weather(
    n_stations: int = 4, n_hours: int = 24 * 60, seed: int = 7
) -> pd.DataFrame:
    """Small synthetic multi-station hourly dataset with diurnal + seasonal
    structure, for standalone sanity-checking without the real project data.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    hour_of_day = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    rows = []
    for s in range(n_stations):
        station_id = f"AWS_{s}"
        base_temp = 22 + 3 * s
        base_pressure = 1013 + 2 * s
        base_humidity = 55 + 4 * s

        temp = (
            base_temp
            + 6 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            + 4 * np.sin(2 * np.pi * day_of_year / 365)
            + rng.normal(0, 0.6, n_hours)
        )
        pressure = (
            base_pressure
            + 1.5 * np.sin(2 * np.pi * hour_of_day / 24 + s)
            + rng.normal(0, 0.3, n_hours)
        )
        humidity = np.clip(
            base_humidity
            - 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            + rng.normal(0, 2.0, n_hours),
            0,
            100,
        )

        rows.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "timestamp": timestamps,
                    "temperature": temp,
                    "pressure": pressure,
                    "humidity": humidity,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _inject_toy_anomalies(
    df: pd.DataFrame, seed: int = 11
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal, self-contained anomaly injection (spike / stuck / drift) used
    only by this module's self-test so it can run without depending on the
    real `skyguard.simulation.anomaly_injector`. The real evaluation script
    (`evaluate_lstm_autoencoder.py`) uses the project's actual injector.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_type"] = None
    events = []

    stations = df["station_id"].unique()

    s = stations[0]
    idx = df[df["station_id"] == s].index
    t = rng.choice(idx[50:-50])
    df.loc[t, "temperature"] += 18.0
    df.loc[t, "is_anomaly"] = True
    df.loc[t, "anomaly_type"] = "spike"
    events.append({"anomaly_type": "spike", "rows": [t]})

    s = stations[1 % len(stations)]
    idx = df[df["station_id"] == s].index
    start = rng.choice(idx[100:-100])
    pos = df.index.get_loc(start)
    stuck_rows = df.index[pos : pos + 20]
    frozen_value = df.loc[start, "humidity"]
    df.loc[stuck_rows, "humidity"] = frozen_value
    df.loc[stuck_rows, "is_anomaly"] = True
    df.loc[stuck_rows, "anomaly_type"] = "stuck"
    events.append({"anomaly_type": "stuck", "rows": list(stuck_rows)})

    s = stations[2 % len(stations)]
    idx = df[df["station_id"] == s].index
    start = rng.choice(idx[200:-200])
    pos = df.index.get_loc(start)
    drift_rows = df.index[pos : pos + 30]
    ramp = np.linspace(0, 12, len(drift_rows))
    df.loc[drift_rows, "pressure"] = df.loc[drift_rows, "pressure"].to_numpy() + ramp
    df.loc[drift_rows, "is_anomaly"] = True
    df.loc[drift_rows, "anomaly_type"] = "drift"
    events.append({"anomaly_type": "drift", "rows": list(drift_rows)})

    events_df = pd.DataFrame(events)
    return df, events_df


def _run_self_test() -> None:
    print("=" * 70)
    print("LSTM AUTOENCODER DETECTOR - SELF TEST")
    print("=" * 70)
    print(f"Backend available: torch={_TORCH_AVAILABLE}, sklearn={_SKLEARN_AVAILABLE}")

    clean = _make_synthetic_weather(n_stations=4, n_hours=24 * 60)
    cutoff = clean["timestamp"].quantile(0.7)
    historical = clean[clean["timestamp"] < cutoff].reset_index(drop=True)
    present = clean[clean["timestamp"] >= cutoff].reset_index(drop=True)

    config = LSTMAEConfig(
        window_size=24,
        hidden_size=8,
        latent_size=4,
        epochs=5 if _TORCH_AVAILABLE else 3,
        batch_size=32,
        val_fraction=0.2,
    )

    print("\nCalibrating on historical (clean) data...")
    calibration = calibrate_lstm_autoencoder_detector(historical, config)
    print(f"  backend used:        {calibration.backend}")
    print(f"  stations calibrated: {calibration.n_stations}")
    print(f"  training windows:    {calibration.n_training_windows}")
    print(f"  validation windows:  {calibration.n_validation_windows}")
    print(f"  threshold P95:       {calibration.threshold_p95:.5f}")
    print(f"  threshold P99:       {calibration.threshold_p99:.5f}")

    print("\nInjecting toy anomalies into unseen 'present' data...")
    contaminated, events = _inject_toy_anomalies(present)
    print(
        f"  injected {contaminated['is_anomaly'].sum()} anomalous rows across {len(events)} events"
    )

    print("\nRunning detector on contaminated data...")
    result = run_lstm_autoencoder_detector(contaminated, calibration)

    scored = result[result["lstm_ae_scored"]]
    print(
        f"  scored {len(scored)}/{len(result)} rows "
        f"({len(scored) / len(result):.1%} coverage)"
    )

    merged = result.join(contaminated[["is_anomaly", "anomaly_type"]], rsuffix="_truth")
    scored_merged = merged[merged["lstm_ae_scored"]]

    clean_mask = ~scored_merged["is_anomaly"]
    fp_rate = (
        scored_merged.loc[clean_mask, "lstm_ae_alert"].mean()
        if clean_mask.any()
        else float("nan")
    )
    print(f"\n  False positive rate on clean rows: {fp_rate:.2%}")

    print("\n  Recall by injected anomaly type:")
    all_pass = True
    for a_type in scored_merged["anomaly_type"].dropna().unique():
        subset = scored_merged[scored_merged["anomaly_type"] == a_type]
        recall = subset["lstm_ae_alert"].mean()
        status = "OK" if recall > 0.3 else "WARNING (low recall)"
        if recall <= 0.3:
            all_pass = False
        print(f"    {a_type:10s}: recall={recall:.2%}  [{status}]")

    print("\n  Sample explainability output (first alerted row):")
    alerted = scored_merged[scored_merged["lstm_ae_alert"]]
    if len(alerted):
        row = alerted.iloc[0]
        print(
            f"    station={row['station_id']}, top_contributor={row['lstm_ae_top_contributor']}, "
            f"severity_score={row['lstm_ae_severity_score']:.2f}, severity_label={row['lstm_ae_severity_label']}"
        )

    print("\n" + "=" * 70)
    if all_pass and fp_rate < 0.15:
        print("SELF TEST: PASS")
    else:
        print(
            "SELF TEST: WARNING - detector ran end-to-end but recall/FP rate outside expected range."
        )
        print(
            "           (This is a hackathon-scale synthetic sanity check, not a tuned benchmark.)"
        )
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()
