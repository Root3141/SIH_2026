# SkyGuard Developer Guide

This guide is for developers working on the SkyGuard project.

Its purpose is simple:

> After cloning the repository, you should be able to understand the project architecture, data flow, detector interfaces, evaluation methodology, and current development workflow without needing additional explanation.

SkyGuard is a hackathon project, so this documentation prioritizes practical clarity and reproducibility over production-level process.

---

# 1. What is SkyGuard?

SkyGuard is a multi-detector weather station anomaly detection system.

The project analyzes observations from multiple Automatic Weather Stations (AWS) and attempts to identify faulty, suspicious, or anomalous sensor readings.

The primary monitored variables are:

* Temperature
* Atmospheric Pressure
* Relative Humidity

SkyGuard deliberately uses multiple complementary anomaly detectors rather than relying on a single model.

Different anomaly types have different signatures:

* A sudden spike may be obvious statistically.
* A station behaving differently from nearby stations may be spatially anomalous.
* A subtle temporal pattern may only be visible through sequence modelling.

The architecture therefore combines multiple independent sources of evidence.

---

# 2. Current System Architecture

The current ML architecture consists of three detectors.

```text
                         Historical Data
                         (2023–2025)
                               │
                               ▼
                    Calibration / Training
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Statistical         Spatial          LSTM Autoencoder
         Detector          Detector            Detector
              │                │                │
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                         Fusion Layer
                         (Next Stage)
                               │
                               ▼
                    Final Anomaly Decision
```

## Current detector responsibilities

### Statistical detector

Focuses on deviations from expected temporal and historical behaviour.

Useful for:

* Extreme values
* Sudden deviations
* Historical range violations
* Temporal abnormalities
* Persistence-based behaviour

---

### Spatial detector

Compares a station with neighbouring stations.

Uses spatial interpolation and neighbour consistency.

Useful for:

* Localized station faults
* Isolated offsets
* Spatial inconsistencies
* Station-specific spikes

---

### LSTM Autoencoder

Learns normal multivariate temporal patterns and measures reconstruction error.

Useful for:

* Multivariate temporal anomalies
* Complex temporal patterns
* Drift
* Noise
* Rate changes
* Patterns not easily captured by manually designed statistical rules

The LSTM Autoencoder is intended to complement the existing detectors, not replace them.

---

# 3. Repository Branches and Development Workflow

The repository currently uses separate branches for major development areas.

```text
main
│
├── dev/ml-engine
│   └── Detection algorithms, evaluation, simulation, calibration
│       LSTM Autoencoder and future fusion work
│
└── dev/frontend
    └── Streamlit dashboard and frontend development
```

## `main`

The stable project branch.

Features should be tested before merging.

---

## `dev/ml-engine`

The primary branch for anomaly detection work.

Typical responsibilities include:

* Statistical anomaly detection
* Spatial anomaly detection
* LSTM Autoencoder
* Synthetic anomaly generation
* Detector calibration
* Detector evaluation
* Threshold diagnostics
* Combined detector evaluation
* Detector fusion

---

## `dev/frontend`

Frontend development branch.

Typical responsibilities include:

* Streamlit dashboard
* Alert visualization
* Weather station visualization
* Sensor health displays
* Maps
* Interactive monitoring
* Future detector output integration

---

## General workflow

Developers should generally work within the branch relevant to their component.

```text
ML Developer
     │
     ▼
dev/ml-engine
```

```text
Frontend Developer
     │
     ▼
dev/frontend
```

Avoid creating unnecessary branches for very small changes.

Before beginning significant work:

```bash
git status
git pull
```

Before committing:

```bash
git status
git diff
```

Use descriptive commits.

Examples:

```text
feat: add LSTM autoencoder detector and evaluator
feat: add three-detector combined evaluation
fix: correct LSTM calibration threshold diagnostics
docs: update developer guide for LSTM detector
```

---

# 4. Quick Start

## Clone the ML engine branch

```bash
git clone -b dev/ml-engine https://github.com/Root3141/SIH_2026.git
cd SIH_2026
```

If already cloned:

```bash
git fetch origin
git checkout dev/ml-engine
git pull origin dev/ml-engine
```

---

## Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install -e .
```

The LSTM detector uses PyTorch when available.

---

# 5. Project Structure

The important repository structure is approximately:

```text
SIH_2026/
│
├── README.md
├── DEVELOPER_GUIDE.md
├── CODEBASE_REFERENCE.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── results/
│   ├── statistical/
│   ├── spatial/
│   ├── lstm_autoencoder/
│   └── statistical_spatial_lstm/
│
├── frontend/
│   └── streamlit/
│
└── src/
    └── skyguard/
        ├── config/
        ├── data/
        ├── detectors/
        │   ├── statistical.py
        │   ├── spatial.py
        │   └── lstm_autoencoder.py
        │
        ├── evaluation/
        │   ├── evaluate_statistical.py
        │   ├── evaluate_spatial.py
        │   ├── evaluate_lstm_autoencoder.py
        │   └── evaluate_combined.py
        │
        ├── fusion/
        │   └── __init__.py
        │
        └── simulation/
            └── anomaly_injector.py
```

---

# 6. Data Architecture

SkyGuard currently separates data chronologically.

## Historical calibration data

Location:

```text
data/raw/ncr_weather_historical.parquet
```

Period:

```text
2023-01-01 through 2025-12-31
```

Purpose:

* Detector calibration
* Threshold estimation
* Normal behaviour modelling
* ML training
* Validation

Historical data must not be mixed with unseen evaluation data during calibration.

---

## Unseen evaluation data

Location:

```text
data/raw/ncr_weather_2026_present.parquet
```

Current period:

```text
2026-01-01 onward
```

Purpose:

* Unseen detector evaluation
* Synthetic anomaly injection
* Benchmark experiments

The intended workflow is:

```text
Historical Data
      │
      ▼
Calibration / Training
      │
      │
      └───────────────┐
                      │
                      ▼
              Frozen Detector State


Unseen Evaluation Data
      │
      ▼
Synthetic Anomaly Injection
      │
      ▼
Run Frozen Detector
      │
      ▼
Evaluate Predictions
```

This separation is important for preventing evaluation leakage.

---

# 7. Synthetic Anomaly Injection

Location:

```text
src/skyguard/simulation/anomaly_injector.py
```

The anomaly injector creates controlled synthetic sensor faults.

This allows the project to evaluate detectors against known ground truth.

Typical usage:

```python
from skyguard.simulation.anomaly_injector import (
    AnomalyConfig,
    inject_anomalies,
)

config = AnomalyConfig()

corrupted_df, injection_log = inject_anomalies(
    clean_df,
    variables,
    config,
)
```

---

## Supported anomaly types

| Anomaly Type | Description                         |
| ------------ | ----------------------------------- |
| spike        | Sudden extreme deviation            |
| offset       | Sustained shift from expected value |
| drift        | Gradual movement away from normal   |
| stuck        | Sensor becomes constant             |
| rate_change  | Abnormally rapid temporal change    |
| noise        | Increased random variation          |
| dropout      | Missing or invalid readings         |

Not every detector is expected to detect every anomaly type equally well.

This is intentional.

---

## Important note about dropout

Dropout is fundamentally different from value-based anomalies.

A missing or invalid reading may not be suitable for reconstruction-based or normal numerical anomaly scoring.

Therefore:

> Detector comparisons should not be judged harshly for poor dropout detection unless explicit missing-data detection logic is part of that detector.

Dropout should eventually be handled by dedicated data-quality or missing-value logic.

---

# 8. Evaluation Philosophy

The project evaluates detectors using a controlled synthetic benchmark.

The standard process is:

```text
                 Historical Dataset
                        │
                        ▼
                 Calibrate Detector
                        │
                        ▼
              Freeze Calibration State
                        │
                        ▼
              Unseen Evaluation Dataset
                        │
                        ▼
              Inject Known Anomalies
                        │
                        ▼
                Run Detector Once
                        │
                        ▼
              Compare with Ground Truth
                        │
                        ▼
              Calculate Performance
```

Synthetic labels are used for evaluation only.

They must not be used to:

* Train the detector
* Recalibrate thresholds
* Tune the detector during normal inference

They may be used for offline experiments and threshold analysis.

---

# 9. Observation-Level Metrics

Each row is evaluated independently.

Definitions:

### True Positive

An anomalous observation was correctly flagged.

### False Positive

A clean observation was incorrectly flagged.

### False Negative

An anomalous observation was missed.

### True Negative

A clean observation was correctly left unflagged.

---

## Precision

```text
TP
─────────
TP + FP
```

Of everything flagged, how much was actually anomalous?

---

## Recall

```text
TP
─────────
TP + FN
```

Of all anomalies, how many were detected?

---

## F1 Score

```text
2 × Precision × Recall
──────────────────────
   Precision + Recall
```

Balances precision and recall.

---

## False Positive Rate

```text
FP
─────────
FP + TN
```

Operationally, this is important because excessive false positives cause alert fatigue.

---

# 10. Event-Level Evaluation

Many anomalies span multiple observations.

Example:

```text
Hour 1 → anomaly
Hour 2 → anomaly
Hour 3 → anomaly
Hour 4 → anomaly
```

Observation-level evaluation asks:

> How many anomalous rows were detected?

Event-level evaluation asks:

> Was the anomaly event detected at least once?

For an event:

```text
Detected if:

at least one anomalous observation
within the event is flagged
```

Event-level recall is:

```text
Detected Events
────────────────────
Total Anomaly Events
```

Event-level evaluation is particularly useful for operational anomaly detection.

Catching one point in a meaningful anomaly event may be enough to trigger investigation.

---

# 11. Window Overlap Contamination

Window-based detectors require additional care during evaluation.

The LSTM Autoencoder uses overlapping temporal windows.

An anomaly can therefore influence reconstruction scores of nearby observations.

For a window of length 24:

```text
        ┌────── 24-hour window ──────┐
        │                             │
normal normal anomaly anomaly normal normal
        │                             │
        └──── overlapping windows ────┘
```

A clean row close to an anomaly may receive a higher reconstruction error because it belongs to windows containing anomalous observations.

Therefore, clean observations are separated into:

### Near-anomaly clean rows

Clean observations close enough to anomalies to potentially share windows.

### Buffer-clear clean rows

Clean observations sufficiently far from anomalies.

For the LSTM:

```text
window size = 24
overlap buffer = ±23 observations
```

Buffer-clear rows provide a cleaner estimate of intrinsic false-positive behaviour.

---

# 12. Statistical Detector

Location:

```text
src/skyguard/detectors/statistical.py
```

The statistical detector identifies observations that are unusual relative to expected temporal and historical behaviour.

Typical signals include:

* Range violations
* Rate-of-change anomalies
* Z-score deviations
* Persistence signals
* Historical deviations

Run evaluation:

```bash
python src/skyguard/evaluation/evaluate_statistical.py
```

Results are saved under:

```text
results/statistical/
```

The statistical detector is generally strong for obvious rule-based and temporal deviations.

---

# 13. Spatial Detector

Location:

```text
src/skyguard/detectors/spatial.py
```

The spatial detector compares station observations with nearby stations.

Conceptually:

```text
Station A observation: 35°C

Nearby stations:
B: 22°C
C: 23°C
D: 21°C

Expected spatial value ≈ 22°C

Residual:

35 - 22 = 13°C

Large residual → possible anomaly
```

The detector uses spatial relationships and Inverse Distance Weighting (IDW).

Current evaluated configuration includes:

```text
k neighbors:          4
IDW power:            2.0
suspicious percentile: 95
anomaly percentile:    99.5
```

Run evaluation:

```bash
python src/skyguard/evaluation/evaluate_spatial.py
```

Results:

```text
results/spatial/
```

---

# 14. LSTM Autoencoder Detector

Location:

```text
src/skyguard/detectors/lstm_autoencoder.py
```

The LSTM Autoencoder is SkyGuard's deep temporal anomaly detector.

Its purpose is to learn normal multivariate temporal behaviour across:

* Temperature
* Pressure
* Humidity

Rather than manually specifying every possible anomaly pattern, the model learns to reconstruct normal sequences.

Unusual sequences should produce larger reconstruction errors.

---

## Core concept

Training:

```text
Normal Time-Series Windows
          │
          ▼
      LSTM Encoder
          │
          ▼
      Latent State
          │
          ▼
      LSTM Decoder
          │
          ▼
Reconstructed Window
```

Anomaly score:

```text
Original Window
       -
Reconstructed Window
       │
       ▼
Reconstruction Error
```

Higher reconstruction error indicates greater deviation from learned normal behaviour.

---

## Key design decisions

The implementation follows several important constraints.

### Station boundaries are respected

Windows are constructed independently for each station.

The detector does not create windows that mix:

```text
Station A → Station B
```

---

### Large gaps are not bridged

Temporal windows should not cross discontinuous periods.

---

### Chronological separation

The LSTM follows a chronological workflow:

```text
Historical Data
      │
      ▼
Training Slice
      │
      ▼
Train Model
      │
      ▼
Validation Slice
      │
      ▼
Calibrate Thresholds
```

The unseen 2026 evaluation data is not used to train the model.

---

### Multivariate modelling

The model sees all primary variables together:

```text
temperature
pressure
humidity
```

This allows the detector to learn relationships between variables.

---

### Observation-level scoring

The model reconstructs temporal windows, but SkyGuard evaluates observations.

Overlapping window reconstruction errors are therefore aggregated back into observation-level scores.

This allows the LSTM detector to integrate with the same evaluation framework as the other detectors.

---

# 15. LSTM Calibration and Cached State

The LSTM evaluator supports caching.

Primary artifacts:

```text
results/lstm_autoencoder/
├── calibration.pkl
└── scored_evaluation.parquet
```

---

## `calibration.pkl`

Contains the frozen detector state.

This includes:

* Trained model
* Normalization state
* Configuration
* Calibration thresholds
* Relevant calibration metadata

The calibration state should be reused when evaluating the same trained model.

---

## `scored_evaluation.parquet`

Contains evaluation data after LSTM inference.

This allows diagnostics such as:

* Threshold sweeps
* Score distributions
* Buffer analysis
* False-positive analysis

without retraining the model.

---

# 16. LSTM Evaluator Modes

Location:

```text
src/skyguard/evaluation/evaluate_lstm_autoencoder.py
```

The evaluator supports three modes.

---

## Full mode

```bash
python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode full
```

Workflow:

```text
Load Historical Data
        │
        ▼
Train / Calibrate LSTM
        │
        ▼
Save calibration.pkl
        │
        ▼
Inject anomalies
        │
        ▼
Run inference
        │
        ▼
Evaluate
        │
        ▼
Save scored results
```

Use this after significant model changes.

---

## Score mode

```bash
python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode score
```

Workflow:

```text
Load Existing calibration.pkl
        │
        ▼
Inject Fresh Evaluation Anomalies
        │
        ▼
Run LSTM Inference
        │
        ▼
Evaluate
```

Important:

> Score mode does not retrain the LSTM.

It reuses the stored calibration and trained weights.

---

## Analyze mode

```bash
python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode analyze
```

Workflow:

```text
Load calibration.pkl
        │
        ▼
Load scored_evaluation.parquet
        │
        ▼
Run Diagnostics
```

No retraining.

No LSTM inference.

This mode is useful for cheap analysis experiments.

---

# 17. LSTM Thresholds

The LSTM calibration currently produces percentile-based thresholds.

Important values include:

```text
Calibration P95
Calibration P99
```

The detector currently uses the calibrated P99 threshold for confirmed alerts.

Conceptually:

```text
score >= P95
        │
        ▼
Suspicious

score >= P99
        │
        ▼
Confirmed anomaly
```

The threshold should remain a calibration-derived quantity.

Evaluation data may be used to study threshold tradeoffs, but evaluation-derived thresholds should not silently replace production calibration thresholds.

---

# 18. LSTM Diagnostic Findings

The LSTM evaluation showed strong event-level performance for several anomaly types.

The detector was particularly effective for:

* Drift
* Noise
* Rate change
* Spikes
* Many offsets

It was weaker for:

* Stuck anomalies
* Dropouts

Dropout should not currently be treated as a primary LSTM performance target because missing-data detection is a separate problem.

Threshold increases reduce false positives but primarily begin sacrificing:

* Stuck detection
* Some offsets
* Some drift events

The detector therefore currently remains unchanged pending fusion experiments.

---

# 19. Why the LSTM Is Not Being Further Tuned Yet

The LSTM has reached a useful stage where further isolated tuning has diminishing returns.

Combined evaluation showed that the LSTM contributes complementary detections, particularly events missed by the existing statistical and spatial detectors.

However, it also produces additional false positives when used independently.

Therefore, the next optimisation problem is no longer:

> How do we make the LSTM perfect by itself?

It is:

> How do we combine complementary detector evidence intelligently?

This is the responsibility of the upcoming fusion layer.

The LSTM should currently be treated as a frozen detector baseline.

Do not casually retrain or change thresholds before fusion experiments establish whether such changes improve the final system.

---

# 20. Combined Three-Detector Evaluation

Location:

```text
src/skyguard/evaluation/evaluate_combined.py
```

The combined evaluator runs:

* Statistical detector
* Spatial detector
* LSTM Autoencoder

against the same synthetic benchmark.

This is critical.

Detector comparisons are only meaningful when they use:

```text
Same evaluation dataset
Same injected anomalies
Same anomaly seed
Same ground truth
```

---

## Combined evaluation principle

```text
                    Shared Synthetic Benchmark
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Statistical          Spatial            LSTM
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                       Fusion Experiments
```

The combined evaluator does not retrain the LSTM.

It loads the existing frozen calibration state.

---

# 21. Fusion Baselines

The combined evaluator currently tests simple logical combinations.

Examples:

### Statistical + Spatial

```text
statistical OR spatial
```

### Statistical + LSTM

```text
statistical OR LSTM
```

### Spatial + LSTM

```text
spatial OR LSTM
```

### All detectors

```text
statistical OR spatial OR LSTM
```

---

## Any two of three

```text
statistical + spatial + LSTM

Alert if at least two agree
```

Conceptually:

```text
S + P = alert
S + L = alert
P + L = alert
```

This often provides a useful precision-oriented baseline.

---

## All three

```text
statistical AND spatial AND LSTM
```

This is extremely conservative.

It is expected to produce:

* High precision
* Very low false-positive rates
* Lower recall

---

# 22. LSTM Complementarity

The purpose of adding the LSTM is not necessarily to beat every existing detector individually.

The important question is:

> Does the LSTM detect meaningful anomalies that the other detectors miss?

This is measured through complementarity analysis.

For example:

```text
Baseline detectors detect:
117 events

Adding LSTM detects:
124 events

New events recovered by LSTM:
7
```

This indicates that the LSTM contributes unique information.

However, complementarity must always be considered alongside false-positive cost.

A detector that recovers a few events but introduces thousands of unique false positives should not automatically be OR-combined without further fusion logic.

---

# 23. Current Detector Status

## Statistical Detector

Status:

```text
ACTIVE BASELINE
```

Strength:

* Good precision
* Strong general-purpose rule-based detection

---

## Spatial Detector

Status:

```text
ACTIVE BASELINE
```

Strength:

* Geographic consistency
* Localized station faults

---

## LSTM Autoencoder

Status:

```text
FROZEN BASELINE
```

Current state:

* Trained calibration cached
* No retraining required for evaluation
* P99 baseline established
* Threshold multiplier diagnostics completed
* Persistence simulation tested
* Combined complementarity evaluated

The detector should remain unchanged until fusion experiments indicate a clear reason for modification.

---

# 24. Results Directories

Current evaluation outputs are separated by experiment.

```text
results/
│
├── statistical/
│
├── spatial/
│
├── lstm_autoencoder/
│   ├── calibration.pkl
│   ├── scored_evaluation.parquet
│   ├── threshold_sweep.csv
│   ├── lstm_ae_evaluation.png
│   └── lstm_ae_threshold_analysis.png
│
└── statistical_spatial_lstm/
    ├── combined_metrics.csv
    ├── combined_full_results.parquet
    └── three_detector_comparison.png
```

Generated results should generally not be manually edited.

They should be reproducible by rerunning the relevant evaluation pipeline.

---

# 25. Detector Output Conventions

Detector output columns should use a consistent prefix.

Examples:

```text
statistical_alert
spatial_alert
lstm_ae_alert
```

LSTM-specific outputs include concepts such as:

```text
lstm_ae_score
lstm_ae_severity_score
lstm_ae_severity_label
lstm_ae_alert
lstm_ae_top_contributor
```

Per-variable reconstruction diagnostics may also be produced:

```text
temperature_lstm_ae_error
pressure_lstm_ae_error
humidity_lstm_ae_error
```

New detectors should follow the same convention:

```text
detector_name_score
detector_name_alert
detector_name_severity_label
```

Avoid generic names such as:

```text
score
alert
prediction
```

because multiple detector outputs coexist in combined evaluation.

---

# 26. Adding a New Detector

A new detector should generally follow this structure.

## Step 1: Add detector implementation

Location:

```text
src/skyguard/detectors/
```

Example:

```text
my_detector.py
```

---

## Step 2: Define calibration

Separate:

```text
Calibration / Training
```

from:

```text
Inference
```

The detector should not silently learn from evaluation data.

---

## Step 3: Define stable output columns

Example:

```text
my_detector_score
my_detector_alert
my_detector_severity
```

---

## Step 4: Create an evaluator

Location:

```text
src/skyguard/evaluation/
```

Example:

```text
evaluate_my_detector.py
```

The evaluator should:

1. Load historical calibration data
2. Calibrate/train using historical data only
3. Load unseen evaluation data
4. Inject anomalies
5. Run inference
6. Calculate observation metrics
7. Calculate event metrics
8. Evaluate by anomaly type
9. Analyze false positives
10. Save artifacts

---

## Step 5: Test complementarity

Do not stop at standalone metrics.

Also ask:

```text
Which anomalies does this detector uniquely detect?
```

A detector with moderate standalone metrics may still be valuable if it detects anomaly classes missed by existing detectors.

---

# 27. Fusion Development

The next major development area is:

```text
src/skyguard/fusion/
```

Fusion should consume detector outputs rather than modifying detector internals.

Conceptually:

```text
Statistical Output
        │
Spatial Output
        │
LSTM Output
        │
        ▼
   Fusion Engine
        │
        ▼
Final Score / Alert
```

The fusion layer should initially treat detectors as independent evidence sources.

---

## Recommended initial fusion experiments

Start with simple baselines.

### Experiment 1

```text
Statistical OR Spatial
```

Current baseline.

---

### Experiment 2

```text
Any 2 of 3 detectors
```

Useful for reducing false positives while maintaining complementary evidence.

---

### Experiment 3

Weighted score

Example:

```text
final_score =
    w_statistical × statistical_score
  + w_spatial × spatial_score
  + w_lstm × lstm_score
```

Scores should be normalized before combining.

---

### Experiment 4

Evidence-aware fusion

Example:

```text
High statistical confidence
+ High spatial confidence

→ Strong anomaly

Moderate LSTM anomaly
+ Statistical confirmation

→ Strong anomaly

LSTM-only anomaly

→ Suspicious / lower confidence
```

---

# 28. Fusion Evaluation Rules

Fusion experiments must not be evaluated casually.

Every fusion experiment should use:

```text
Same benchmark
Same anomaly injection
Same random seed
Same evaluation metrics
```

Track:

* Precision
* Recall
* F1
* False-positive rate
* Buffer-clear false-positive rate
* Event recall
* Recall by anomaly type
* Unique events recovered

Do not optimise only for F1.

SkyGuard is an anomaly detection system.

Operational considerations matter:

```text
Too many alerts
        ↓
Alert fatigue

Too few alerts
        ↓
Missed faults
```

The correct operating point depends on the intended final use case.

---

# 29. Recommended Development Order

The current recommended development sequence is:

```text
1. Statistical Detector
        ✓ baseline complete

2. Spatial Detector
        ✓ baseline complete

3. LSTM Autoencoder
        ✓ baseline complete

4. Combined Evaluation
        ✓ baseline complete

5. Detector Fusion
        ← CURRENT NEXT STEP

6. Final Integrated Evaluation

7. Backend / API Integration

8. Frontend Integration
```

The current priority is fusion.

Further isolated LSTM optimisation should be postponed unless fusion experiments reveal a specific weakness that requires detector-level changes.

---

# 30. Important Experimental Rules

## Do not retrain accidentally

If evaluating the existing LSTM baseline:

```text
Load calibration.pkl
```

Do not retrain unless explicitly testing a new model configuration.

---

## Do not compare detectors on different anomaly injections

Bad:

```text
Statistical → seed A
Spatial → seed B
LSTM → seed C
```

Good:

```text
Shared benchmark → all detectors
```

---

## Do not tune on test labels without documenting it

Synthetic evaluation labels are useful for experiments.

However, thresholds selected directly from evaluation data should be clearly identified as experimental.

Production thresholds should originate from calibration data unless a formal validation procedure is introduced.

---

## Distinguish clean false positives from window contamination

For window-based detectors, use buffer-aware analysis.

Do not assume every clean row near an anomaly represents an intrinsic detector false positive.

---

## Keep dropout separate

Dropout is primarily a missing-data problem.

Do not distort detector comparisons by expecting every numerical anomaly detector to solve it.

---

# 31. Reproducibility Checklist

Before reporting an experiment, record:

```text
Detector configuration
Calibration dataset period
Evaluation dataset period
Random seed
Anomaly injection rate
Anomaly types included
Threshold configuration
Fusion policy
Persistence policy
```

For combined experiments also record:

```text
Was the benchmark shared? YES/NO
Was the LSTM retrained? YES/NO
Was inference run once? YES/NO
Were detector outputs reused? YES/NO
```

---

# 32. Current Architecture Summary

SkyGuard currently has three complementary anomaly detectors.

```text
STATISTICAL
    │
    ├── Historical / temporal deviations
    │
SPATIAL
    │
    ├── Neighbour consistency
    │
LSTM AUTOENCODER
    │
    ├── Multivariate temporal reconstruction
    │
    ▼
FUSION
    │
    ▼
FINAL ANOMALY DECISION
```

The current philosophy is:

> Do not force every detector to solve every anomaly type.

Instead:

> Build complementary detectors, evaluate them fairly on the same benchmark, measure what unique information each contributes, and combine their evidence intelligently.

---

# 33. Current Development Status

| Component                          | Status                |
| ---------------------------------- | --------------------- |
| Data pipeline                      | Active                |
| Synthetic anomaly injector         | Active                |
| Statistical detector               | Baseline complete     |
| Spatial detector                   | Baseline complete     |
| LSTM Autoencoder                   | Baseline frozen       |
| LSTM threshold diagnostics         | Completed             |
| LSTM persistence simulation        | Tested                |
| Three-detector combined evaluation | Completed             |
| Complementarity analysis           | Completed             |
| Fusion layer                       | Next development task |
| Backend integration                | Future                |
| Frontend integration               | Future                |

---

# Final Guidance for Developers

When making changes, follow this principle:

```text
Change one thing
      ↓
Evaluate it reproducibly
      ↓
Compare against baseline
      ↓
Keep it only if it improves the system
```

Do not optimise detectors in isolation indefinitely.

SkyGuard is a multi-detector anomaly detection system.

The value of a detector is determined not only by its standalone performance, but also by:

* What anomalies it uniquely detects
* Whether it complements other detectors
* How many false positives it introduces
* How useful its evidence is to the fusion layer

The current next step is therefore to build and evaluate detector fusion using the frozen Statistical, Spatial, and LSTM Autoencoder baselines.
