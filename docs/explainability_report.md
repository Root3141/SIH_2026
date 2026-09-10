# SkyGuard explainability integration

Validated on branch `kashika-final-integration`, starting from clean commit `c0dbedf`.
No commit or push performed. Initial worktree and SHA-256 snapshots are stored beside this report.

## Architecture and boundaries

`run_fusion(raw weather)` → unchanged frozen detector votes → `enrich_evidence(fused)` →
`explain_dataframe(...)`. Each explanation contains four separate sections:

- **Decision:** the actual deterministic 2-of-3 detector votes, required agreement, final alert,
  severity and agreement fraction. Inconsistent votes/final-alert inputs are rejected rather than explained falsely.
- **Detector evidence:** existing statistical checks, spatial neighbor comparisons, and reconstruction outputs.
- **Model explanation:** optional SHAP contributions to the scalar reconstruction anomaly score.
- **Operational diagnosis:** the existing Phase-1 hypotheses, supporting evidence and suggested checks.
  Confidence is a heuristic evidence grade, never a calibrated failure probability.

SHAP does not explain fusion: counting the three Boolean votes is the complete software decision path.
`MINIMUM_VOTES`, severity mapping, weights, detector thresholds, injection and benchmark generation remain unchanged.
Fusion, detector and simulation sources and existing model/data artifacts match all 22 protected hashes.
The only diagnosis source adjustment makes `surface_pressure` optional because current fusion requires
three core variables. Available surface-pressure readings still participate; absent columns are never
fabricated as dropout. Rule constants and precedence are unchanged; all original diagnosis tests pass.

## Evidence schema and compatibility

Current fusion retains statistical summary fields and `lstm_ae_*` summary fields, but omits statistical
per-variable checks and `{variable}_lstm_ae_*` reconstruction evidence. Rather than change production
fusion output, `enrich_evidence` runs the existing frozen statistical/reconstruction detector functions
on the same raw observations, aligns on station/timestamp, asserts equality for all overlapping fields,
and appends only missing columns. This incurs one extra statistical and reconstruction inference pass.
It does not copy or reimplement detector reconstruction logic.

Statistical evidence includes existing hourly residuals, absolute rate of change, range/ROC/persistence
flags and rolling z-scores. Numeric thresholds are not invented where runtime fields omit them.
The statistical primary variable ranks triggered checks and is a display choice, not an extra decision rule.
Spatial evidence includes expected value, signed residual, retained neighbor count/MAD and existing
anomaly/suspicious flags. Humidity evidence is visible but does not drive the current spatial alert.
Reconstruction evidence retains observed, expected, signed residual, normalized squared error, score,
scored status, top contributor and severity. Unscored windows remain explicitly unavailable.

## Actual frozen backend and exact SHAP target

`artifacts/skyguard_v1/lstm.pkl` contains `LSTMAECalibration`, backend **sklearn**, with an
**MLPRegressor** reconstruction model. It is not a frozen PyTorch LSTM. Variable order is
`temperature, pressure, humidity`; each model input is a 24 × 3 window flattened in C order to 72 values.
`model.predict` returns 72 reconstructions reshaped to the same window dimensions.

The wrapper uses `_prepare`, `_normalize_frame` and `_build_windows` from the detector. Normalization
uses frozen station/global means and standard deviations and the existing ±8 clipping. Window stride,
gap segmentation, minimum validity and filling follow the frozen configuration.

The scalar target is the **actual observation score**, not a whole-window average:

1. Select every valid runtime window covering the requested observation.
2. Reconstruct those windows with `_reconstruct_sklearn`.
3. Square normalized input-minus-reconstruction errors.
4. Call `_aggregate_windows_to_observations` with the frozen aggregation setting (`mean`).
5. Average the requested observation's three aggregated variable errors.

The wrapper's unperturbed result must match the stored `lstm_ae_score` within numerical tolerance;
otherwise SHAP falls back. The observation can occur in up to 24 overlapping windows. To keep repeated
readings consistent under perturbation, features are unique timestep/variable positions in their union:
24–47 timestamps × 3 variables (72–141 features), rather than 72 positions from one arbitrary window.
The exact score wrapper also supports the detector helper's `max` aggregation without changing it.
Gappy contexts with window-dependent filling are declined rather than approximated.

## SHAP sampling, aggregation and limits

Uses model-agnostic `shap.PermutationExplainer` (not `LinearExplainer`), following the
[official permutation explainer API](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html).
Default limits are 25 historical background contexts, 2 requested final-alert rows, 2 permutations,
30 seconds per attempt and 500,000 reconstructed model windows per attempt. Configurations have hard
caps on background size (50), requested rows (20) and permutations (10). Runtime is cooperatively
checked between model calls and after SHAP; a native call or import cannot be forcibly interrupted.

Background data are restricted to the chronological historical training portion preceding held-out
validation and all runtime data. Nonoverlapping complete historical contexts of matching length are
sampled with seed 42, without labels or alert-based selection. Cached prepared data are scoped to an
individual request context and requested station; no global cross-artifact/background cache is used.

Outputs include signed timestep-variable contributions with real relative-hour offsets, base value,
target score, top timestep and per-variable `sum(abs(SHAP))` **model contribution magnitude**.
Signed contributions plus the base value must reconstruct the scalar score. The magnitude bars do not
sum to a probability and are not causal importance. Positive offsets explicitly identify future data.

SHAP is lazy and opt-in. Import failure, unsupported backend, missing history, missing/nonfinite
context, score mismatch or exceeded budgets return `shap_available=false`, `shap_status=unavailable`
and a reason. Reconstruction evidence is retained. Nonalerts and alerts outside the requested budget
are counted separately from failed attempts. The production detector/fusion never imports SHAP.

## Diagnosis integration and missing reference limitation

`evaluate_diagnosis.py` now reads the current demo dataset, runs frozen 2-of-3 fusion, maps current
spatial expected/count fields into the diagnosis engine's peer input names, predicts, and only then
joins synthetic labels for evaluation. The obsolete imports and old Phase-2 result dependencies were
removed. Conditional class/family precision, recall, F1, confusion matrices, joint shape-variable
accuracy and overlap-inclusive event consistency/membership metrics are retained.

**The old historical normalization CSV and metadata are absent from this checkout.** No replacement
scales or thresholds were fitted. `--reference <csv> --historical-end <timestamp>` loads an existing
compatible historical reference. Without it, the engine's existing no-reference behavior abstains on
shape rules, while observed nonfinite readings can still support `DATA_DROPOUT`. This fallback is
explicit in every diagnosis object. A supplied reference must match the spatial evidence definition
and strictly precede the evaluation timeline.

All 1,213 final alerts have a diagnosis output, but only
1 has a specific diagnosis (0.0824%).
The remaining alerts are `UNCLASSIFIED_ANOMALY`. This is not a successful validation of fault-shape
classification: among 701 singly labeled detected anomalous rows,
conditional diagnosis accuracy is 0.1427% under the missing-reference fallback.
Do not interpret assignment coverage as useful diagnostic coverage.

## Integrity results and cost

| Check | Result |
|---|---:|
| Fused rows / explanations | 70,848 / 70,848 |
| Final alerts | 1,213 |
| Complete detailed detector evidence | 70,742 |
| Reconstruction evidence | 70,742 |
| Unscored reconstruction rows | 106 |
| Requested SHAP attempts / successful | 2 / 2 |
| SHAP not requested: nonalerts / remaining alerts | 69,635 / 1,211 |
| Mean successful SHAP time | 26.13 seconds |
| Full integrity run (including re-inference and leakage check) | 136.75 seconds |
| Row loss / duplicate keys | 0 / 0 |
| Independent production comparison | 40 protected columns exactly equal over all 70,848 rows |
| Focused / full tests | 49 / 49 passed |

The initial full run had one successful SHAP explanation and one safe timeout. Restricting preparation
to the requested station allowed both requested explanations to complete in the final run. No model
was retrained, and no thresholds or detector decisions were changed to obtain explanations.
Timing is environment-dependent; this is a bounded hackathon demonstration, not a throughput guarantee.

Leakage safeguards use explicit input allowlists before detection, evidence recovery, normalization,
SHAP background construction and diagnosis. Poisoning/removing synthetic and original-value columns
leaves explanations and predictions unchanged. Evaluation alone reads injected labels after prediction.
The pre-existing demo file was absent, so the unchanged `create_final_dataset` entrypoint created it;
existing datasets and benchmark-generation code were preserved.

## Frontend/API and reproduction

`frontend/streamlit/utils/explainability.py` renders the four sections on Station Detail.
It uses current-row vote evidence immediately and optionally reads a single filtered row from
`results/explainability/operator_explanations.parquet`, checking decision/readings before displaying
precomputed detail. Missing optional artifacts retain the basic explanation. No inference or SHAP
computation runs in the UI. `app.py` is unchanged. Dashboard and Station Detail AppTest smoke checks pass.

The operator contract includes `decision`, `detectors`, `model_explanation`, and `diagnosis`, plus
station/timestamp/final-alert identity. The operator sidecar contains no synthetic ground-truth fields.
Full numeric enriched output, integrity metrics, diagnosis metrics and sample alert JSON are saved
alongside this report; generated outputs stay git-ignored, while this report is explicitly unignored.

```python
from skyguard.explainability import enrich_evidence, explain_dataframe
# fused is the existing run_fusion output; reference is optional historical state.
explained = explain_dataframe(enrich_evidence(fused), reference=reference)
operator_object = explained.explanation.iloc[0]
# Pass shap_context=ReconstructionShap(raw, calibration, historical_background)
# explicitly to request bounded detailed model contributions.
```

```sh
PYTHONPATH=src .venv/bin/python -m skyguard.evaluation.evaluate_explainability
PYTHONPATH=src .venv/bin/python -m skyguard.evaluation.evaluate_diagnosis
.venv/bin/python -m pytest -q
```

Batch model evidence can use later overlapping observations. Spike diagnosis requires next-hour
confirmation. These outputs therefore explain the existing retrospective batch decisions; they are
not a claim of causal online diagnosis. Perturbed SHAP contexts can be meteorologically implausible
and correlated features can redistribute attribution. Two permutations are approximate and may be
unstable in fine-grained rankings. Operational classes are hypotheses, not physical fault proof.

## Worked alert

```json
{
  "station_id": "AWS_001",
  "timestamp": "2026-01-01T02:00:00+00:00",
  "decision": {
    "alert_explanation_type": "deterministic_vote",
    "alert_explanation_summary": "2 of 3 detectors agreed, so the final anomaly alert was raised.",
    "detector_vote_summary": {
      "statistical": false,
      "spatial": true,
      "lstm": true
    },
    "statistical_vote": false,
    "spatial_vote": true,
    "lstm_vote": true,
    "agreement_count": 2,
    "agreement_required": 2,
    "final_alert": true,
    "final_severity": "anomaly",
    "final_confidence": 0.6666666666666666,
    "confidence_meaning": "detector agreement fraction, not a calibrated probability"
  },
  "diagnosis": {
    "label": "UNCLASSIFIED_ANOMALY",
    "variable": null,
    "operational_class": "UNCERTAIN",
    "evidence": {
      "candidates": [],
      "confidence_is_probability": false,
      "operational_support": {},
      "precedence": [
        "DATA_DROPOUT",
        "STUCK_SENSOR",
        "NOISY_SENSOR",
        "SPIKE",
        "SENSOR_DRIFT",
        "SENSOR_OFFSET",
        "RATE_CHANGE",
        "UNCLASSIFIED_ANOMALY"
      ],
      "spike_confirmation_lookahead_hours": 1
    },
    "reason": "Available shape evidence does not justify a specific failure mode.",
    "confidence_grade": 0.0,
    "confidence_is_probability": false,
    "reference_status": "historical_reference_unavailable; shape rules abstain",
    "recommendation": "Review available readings and obtain additional evidence."
  },
  "detectors": {
    "statistical": {
      "detector": "statistical",
      "alert": false,
      "primary_variable": "pressure",
      "severity": "suspicious",
      "evidence": {
        "temperature": {
          "temperature_hourly_residual": -10.19625912408759,
          "temperature_rate_of_change": 0.09999999999999964,
          "temperature_range_anomaly": false,
          "temperature_roc_anomaly": false,
          "temperature_persistence_anomaly": false,
          "temperature_zscore_24h_anomaly": false,
          "temperature_zscore_168h_anomaly": false,
          "observed": 11.0
        },
        "pressure": {
          "pressure_hourly_residual": 8.065041217617136,
          "pressure_rate_of_change": 4.3025984025955495,
          "pressure_range_anomaly": false,
          "pressure_roc_anomaly": true,
          "pressure_persistence_anomaly": false,
          "pressure_zscore_24h_anomaly": false,
          "pressure_zscore_168h_anomaly": false,
          "observed": 1018.0842018015587
        },
        "humidity": {
          "humidity_hourly_residual": 22.58759124087591,
          "humidity_rate_of_change": 0.0,
          "humidity_range_anomaly": false,
          "humidity_roc_anomaly": false,
          "humidity_persistence_anomaly": false,
          "humidity_zscore_24h_anomaly": false,
          "humidity_zscore_168h_anomaly": false,
          "observed": 98.0
        }
      },
      "available": true,
      "summary": "Historical baseline, rate and persistence checks are shown below.",
      "observations": [
        "Pressure triggered the roc check."
      ]
    },
    "spatial": {
      "detector": "spatial",
      "alert": true,
      "primary_variable": "pressure",
      "severity": "anomaly",
      "evidence": {
        "temperature": {
          "expected": 10.799865713930863,
          "residual": 0.20013428606913664,
          "neighbor_count": 4,
          "neighbor_mad": 0.07412999999999974,
          "anomaly": false,
          "suspicious": false,
          "observed": 11.0
        },
        "pressure": {
          "expected": 1015.1471086688598,
          "residual": 2.9370931326989194,
          "neighbor_count": 4,
          "neighbor_mad": 0.14825999999994943,
          "anomaly": true,
          "suspicious": true,
          "observed": 1018.0842018015587
        },
        "humidity": {
          "expected": 99.00000000000003,
          "residual": -1.0000000000000284,
          "neighbor_count": 3,
          "neighbor_mad": 0.0,
          "anomaly": false,
          "suspicious": false,
          "observed": 98.0
        }
      },
      "available": true,
      "summary": "Station readings are compared with nearby stations; humidity does not drive the spatial vote.",
      "observations": [
        "Temperature: observed 11.0, neighbor expectation 10.799865713930863, signed residual 0.20013428606913664, retained neighbors 4.",
        "Pressure: observed 1018.0842018015587, neighbor expectation 1015.1471086688598, signed residual 2.9370931326989194, retained neighbors 4.",
        "Humidity: observed 98.0, neighbor expectation 99.00000000000003, signed residual -1.0000000000000284, retained neighbors 3."
      ]
    },
    "lstm": {
      "detector": "lstm",
      "alert": true,
      "primary_variable": "pressure",
      "severity": "anomaly",
      "evidence": {
        "temperature": {
          "error": 0.004185044504510435,
          "expected": 10.554106640819816,
          "residual": 0.44589335918018413,
          "observed": 11.0
        },
        "pressure": {
          "error": 0.13818741555008657,
          "expected": 1015.3694198025873,
          "residual": 2.7147819989713753,
          "observed": 1018.0842018015587
        },
        "humidity": {
          "error": 0.006398473873755305,
          "expected": 96.48720032088471,
          "residual": 1.5127996791152896,
          "observed": 98.0
        }
      },
      "available": true,
      "summary": "Reconstruction errors describe differences from historically learned patterns; they do not establish physical causality.",
      "score": 0.04959031130945077,
      "scored": true,
      "top_contributor": "pressure",
      "severity_score": 1.0,
      "observations": [
        "Temperature: observed 11.0, reconstructed expectation 10.554106640819816, signed residual 0.44589335918018413, normalized squared error 0.004185044504510435.",
        "Pressure: observed 1018.0842018015587, reconstructed expectation 1015.3694198025873, signed residual 2.7147819989713753, normalized squared error 0.13818741555008657.",
        "Humidity: observed 98.0, reconstructed expectation 96.48720032088471, signed residual 1.5127996791152896, normalized squared error 0.006398473873755305."
      ]
    }
  },
  "model_explanation": {
    "method": "shap_permutation",
    "available": true,
    "shap_available": true,
    "shap_status": "available",
    "target": "observation reconstruction anomaly score",
    "score": 0.04959031130945076,
    "base_value": 0.004690233062218651,
    "contribution_label": "model contribution magnitude",
    "variable_contributions": {
      "temperature": 0.2063678031877185,
      "pressure": 0.25772766701899985,
      "humidity": 0.1754318275731225
    },
    "top_variable": "pressure",
    "background_size": 25,
    "top_timestep": "pressure at t+0h",
    "runtime_seconds": 28.729260916938074,
    "model_windows_evaluated": 23553,
    "context_uses_future": true
  }
}
```
