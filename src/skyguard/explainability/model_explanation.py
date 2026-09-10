"""Existing detector evidence and optional bounded reconstruction-score SHAP."""

import re
import numpy as np
import pandas as pd

VARIABLES = ("temperature", "pressure", "humidity")


def value(row, key):
    x = row.get(key)
    if x is None or (np.isscalar(x) and pd.isna(x)):
        return None
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    return x.item() if isinstance(x, np.generic) else x


def detector_evidence(row):
    statistical = {}
    spatial = {}
    reconstruction = {}
    for v in VARIABLES:
        allowed = [
            f"{v}_{s}"
            for s in [
                "hourly_residual",
                "rate_of_change",
                "range_anomaly",
                "roc_anomaly",
                "persistence_anomaly",
            ]
        ]
        allowed += [
            k for k in row.keys() if re.fullmatch(v + r"_zscore_\d+h(?:_anomaly)?", k)
        ]
        statistical[v] = {
            k: value(row, k) for k in allowed if k in row and value(row, k) is not None
        }
        spatial[v] = {
            s: value(row, f"{v}_spatial_{s}")
            for s in [
                "expected",
                "residual",
                "neighbor_count",
                "neighbor_mad",
                "anomaly",
                "suspicious",
            ]
            if value(row, f"{v}_spatial_{s}") is not None
        }
        reconstruction[v] = {
            s: value(row, f"{v}_lstm_ae_{s}")
            for s in ["error", "expected", "residual"]
            if value(row, f"{v}_lstm_ae_{s}") is not None
        }
        for evidence in (statistical[v], spatial[v], reconstruction[v]):
            if evidence:
                evidence["observed"] = value(row, v)
    stat_primary = max(
        VARIABLES,
        key=lambda v: sum(
            bool(x) for k, x in statistical[v].items() if k.endswith("_anomaly")
        ),
    )
    if not any(
        k.endswith("_anomaly") and x for d in statistical.values() for k, x in d.items()
    ):
        stat_primary = None
    # Humidity evidence is retained, but the current spatial vote excludes humidity.
    spatial_primary = next(
        (v for v in ("temperature", "pressure") if spatial[v].get("anomaly")), None
    )
    result = {}
    for name, prefix, evidence, primary, severity in [
        (
            "statistical",
            "statistical",
            statistical,
            stat_primary,
            "statistical_severity_label",
        ),
        ("spatial", "spatial", spatial, spatial_primary, "spatial_severity_level"),
        (
            "lstm",
            "lstm_ae",
            reconstruction,
            value(row, "lstm_ae_top_contributor"),
            "lstm_ae_severity_label",
        ),
    ]:
        available = any(evidence.values())
        result[name] = dict(
            detector=name,
            alert=bool(row[f"{prefix}_alert"]),
            primary_variable=primary,
            severity=value(row, severity),
            evidence=evidence,
            available=available,
            summary=(
                "Detailed evidence unavailable in this row."
                if not available
                else {
                    "statistical": "Historical baseline, rate and persistence checks are shown below.",
                    "spatial": "Station readings are compared with nearby stations; humidity does not drive the spatial vote.",
                    "lstm": "Reconstruction errors describe differences from historically learned patterns; they do not establish physical causality.",
                }[name]
            ),
        )
    result["lstm"].update(
        score=value(row, "lstm_ae_score"),
        scored=value(row, "lstm_ae_scored"),
        top_contributor=value(row, "lstm_ae_top_contributor"),
        severity_score=value(row, "lstm_ae_severity_score"),
    )
    observations = []
    for v, evidence in statistical.items():
        for key, flag in evidence.items():
            if key.endswith("_anomaly") and flag:
                if "_zscore_" in key:
                    score = evidence.get(key.removesuffix("_anomaly"))
                    observations.append(
                        f"{v.title()} rolling residual z-score {score} exceeded its configured threshold."
                    )
                else:
                    kind = key.removeprefix(v + "_").removesuffix("_anomaly")
                    observations.append(f"{v.title()} triggered the {kind} check.")
    result["statistical"]["observations"] = observations
    result["spatial"]["observations"] = [
        f"{v.title()}: observed {e.get('observed')}, neighbor expectation {e.get('expected')}, "
        f"signed residual {e.get('residual')}, retained neighbors {e.get('neighbor_count')}."
        for v, e in spatial.items()
        if e
    ]
    result["lstm"]["observations"] = [
        f"{v.title()}: observed {e.get('observed')}, reconstructed expectation {e.get('expected')}, "
        f"signed residual {e.get('residual')}, normalized squared error {e.get('error')}."
        for v, e in reconstruction.items()
        if e
    ]
    return result


from dataclasses import dataclass
from time import perf_counter
from skyguard.detectors import lstm_autoencoder as ae


@dataclass(frozen=True)
class ShapConfig:
    background_size: int = 25
    max_rows: int = 2
    permutations: int = 2
    max_seconds: float = 30.0
    max_model_windows: int = 500000
    seed: int = 42

    def __post_init__(self):
        if not (
            1 <= self.background_size <= 50
            and 0 <= self.max_rows <= 20
            and 1 <= self.permutations <= 10
            and self.max_seconds > 0
            and self.max_model_windows > 0
        ):
            raise ValueError("Invalid SHAP computation budget")


def unavailable(reason, seconds=0.0):
    return dict(
        method="shap",
        available=False,
        shap_available=False,
        shap_status="unavailable",
        reason=reason,
        runtime_seconds=seconds,
        variable_contributions={},
    )


def normalized_windows(dataframe, calibration):
    variables = calibration.variables
    # Explicit raw-only boundary, including for background data.
    data = ae._prepare(dataframe[["station_id", "timestamp", *variables]], variables)
    c = calibration.config
    norm = ae._normalize_frame(
        data, variables, calibration.global_stats, calibration.station_stats, c.clip_z
    )
    windows, indices, _ = ae._build_windows(
        norm,
        variables,
        c.window_size,
        c.eval_stride,
        c.min_valid_fraction,
        c.max_fill_run,
        c.gap_tolerance,
    )
    return data, norm, windows, indices


class ObservationScore:
    """Scalar *observation* score over all windows covering a requested row.

    Features are unique normalized timestep/variable readings in the union of
    those windows (up to 47 x 3), keeping overlapping copies consistent.
    Reuse reconstruction and aggregation helpers verbatim. Gappy contexts are
    declined by the caller because filling is window dependent.
    """

    def __init__(
        self, calibration, context_indices, window_indices, target, config, reconstruct
    ):
        self.calibration = calibration
        self.indices = window_indices
        self.target = target
        self.lookup = np.searchsorted(context_indices, window_indices)
        self.length = len(context_indices)
        self.config = config
        self.started = perf_counter()
        self.used = 0
        self.reconstruct = reconstruct

    def __call__(self, flat):
        c = self.calibration
        contexts = np.asarray(flat).reshape(-1, self.length, len(c.variables))
        outputs = []
        for context in contexts:
            if perf_counter() - self.started > self.config.max_seconds:
                raise TimeoutError("SHAP time budget exceeded")
            self.used += len(self.indices)
            if self.used > self.config.max_model_windows:
                raise TimeoutError("SHAP prediction budget exceeded")
            windows = context[self.lookup]
            reconstructed = self.reconstruct(windows)
            error = ae._aggregate_windows_to_observations(
                self.indices,
                (windows - reconstructed) ** 2,
                c.variables,
                c.config.score_aggregation,
            )
            outputs.append(float(error.loc[self.target, c.variables].mean()))
        return np.array(outputs)


class ReconstructionShap:
    """Request-scoped cache: frozen calibration, prepared windows and background.

    Never cached globally across datasets/artifacts. Only requested final alerts
    consume the bounded explanation budget. SHAP import is lazy.
    """

    def __init__(self, dataframe, calibration, background, config=None):
        self.config = config or ShapConfig()
        self.calibration = calibration
        self.dataframe = dataframe
        self.background = background
        self.prepared = {}
        self.background_norm = None
        self.attempts = 0
        self.cache = {}

    def explain(self, row):
        started = perf_counter()
        key = (row["station_id"], pd.Timestamp(row["timestamp"]))
        if key in self.cache:
            return self.cache[key]
        if not row["final_alert"]:
            return unavailable("not_requested_nonalert")
        if self.attempts >= self.config.max_rows:
            return unavailable("row_budget_exceeded")
        self.attempts += 1
        try:
            c = self.calibration
            if c.backend not in ("sklearn", "torch"):
                raise ValueError("unsupported_backend")
            if c.backend == "torch":
                torch_model = ae._rehydrate_torch_model(c)
                reconstruct = lambda windows: ae._reconstruct_torch(
                    torch_model, windows, c.config
                )
            else:
                reconstruct = lambda windows: ae._reconstruct_sklearn(
                    c.model_state, windows
                )
            if key[0] not in self.prepared:
                station_data = self.dataframe.loc[self.dataframe.station_id.eq(key[0])]
                self.prepared[key[0]] = normalized_windows(station_data, c)
            data, norm, windows, indices = self.prepared[key[0]]
            matched = np.flatnonzero(
                data.station_id.eq(key[0]) & data.timestamp.eq(key[1])
            )
            if len(matched) != 1:
                raise ValueError("missing_or_duplicate_row_key")
            target = int(matched[0])
            selected = indices[np.any(indices == target, axis=1)]
            if not len(selected):
                raise ValueError("insufficient_history")
            context_indices = np.unique(selected)
            context = norm.loc[context_indices, c.variables].to_numpy(float)
            if not np.isfinite(context).all():
                raise ValueError("window_dependent_gap_filling_unsupported")
            score = ObservationScore(
                c, context_indices, selected, target, self.config, reconstruct
            )
            x = context.reshape(1, -1)
            actual = float(score(x)[0])
            if not np.isclose(actual, row["lstm_ae_score"], rtol=1e-3, atol=1e-5):
                raise ValueError(
                    f"runtime_score_mismatch: "
                    f"runtime={actual:.12g}, "
                    f'stored={float(row["lstm_ae_score"]):.12g}, '
                    f'delta={actual - float(row["lstm_ae_score"]):.12g}'
                )
            if self.background_norm is None:
                raw = ae._prepare(
                    self.background[["station_id", "timestamp", *c.variables]],
                    c.variables,
                )
                if len(raw) == 0 or raw.timestamp.max() >= data.timestamp.min():
                    raise ValueError("background_must_precede_runtime")
                self.background_norm = ae._normalize_frame(
                    raw, c.variables, c.global_stats, c.station_stats, c.config.clip_z
                )
            # Historical contiguous contexts, sampled without labels or runtime alerts.
            bg, _, _ = ae._build_windows(
                self.background_norm,
                c.variables,
                len(context),
                max(len(context), 1),
                1.0,
                c.config.max_fill_run,
                c.config.gap_tolerance,
            )
            if not len(bg):
                raise ValueError("insufficient_background")
            rng = np.random.default_rng(self.config.seed)
            bg = bg[
                rng.choice(
                    len(bg), min(len(bg), self.config.background_size), replace=False
                )
            ].reshape(-1, x.shape[1])
            import shap

            score.started = (
                started  # Includes preparation/import in cooperative deadline.
            )
            offsets = [
                (data.timestamp.iloc[i] - key[1]).total_seconds() / 3600
                for i in context_indices
            ]
            names = [f"{v} at t{offset:+g}h" for offset in offsets for v in c.variables]
            explainer = shap.PermutationExplainer(
                score, bg, feature_names=names, seed=self.config.seed
            )
            explanation = explainer(
                x,
                max_evals=self.config.permutations * (2 * x.shape[1] + 1),
                silent=True,
            )
            if perf_counter() - started > self.config.max_seconds:
                raise TimeoutError("SHAP time budget exceeded")
            contributions = np.asarray(explanation.values).reshape(context.shape)
            magnitudes = dict(
                zip(c.variables, np.abs(contributions).sum(axis=0).tolist())
            )
            base = float(np.asarray(explanation.base_values).reshape(-1)[0])
            if not np.isfinite(contributions).all() or not np.isclose(
                base + contributions.sum(), actual, rtol=1e-5, atol=1e-7
            ):
                raise ValueError("SHAP additivity check failed")
            result = dict(
                method="shap_permutation",
                available=True,
                shap_available=True,
                shap_status="available",
                target="observation reconstruction anomaly score",
                score=actual,
                base_value=base,
                contribution_label="model contribution magnitude",
                variable_contributions=magnitudes,
                top_variable=max(magnitudes, key=magnitudes.get),
                background_size=len(bg),
                timestep_contributions=[
                    dict(
                        variable=v,
                        offset_hours=offset,
                        contribution=float(contributions[i, j]),
                    )
                    for i, offset in enumerate(offsets)
                    for j, v in enumerate(c.variables)
                ],
                top_timestep=names[int(np.abs(contributions).argmax())],
                runtime_seconds=perf_counter() - started,
                model_windows_evaluated=score.used,
                context_uses_future=any(o > 0 for o in offsets),
            )
        except Exception as exc:
            result = unavailable(
                f"{type(exc).__name__}: {exc}", perf_counter() - started
            )
        self.cache[key] = result
        return result
