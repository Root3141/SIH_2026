# SkyGuard Developer Guide

This guide is for anyone joining the SkyGuard project.

Its purpose is simple:

> After cloning the repository, you should be able to understand the project structure, know what each major component does, and run the existing pipelines without needing additional explanation.

SkyGuard is a hackathon project, so this documentation prioritizes practical usability over production-level documentation.

---

## 1. What is SkyGuard?

SkyGuard is a weather station anomaly detection system.

The project analyzes weather observations from multiple Automatic Weather Stations (AWS) and attempts to identify faulty, suspicious, or anomalous sensor readings.

The system uses multiple complementary detection approaches rather than relying on a single model.

Current architecture:

```text
                         Historical Data
                        (2023 - 2025)
                               │
                               ▼
                         Calibration / Training
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Statistical       Spatial Detector    Future ML
         Detector           (IDW-based)       Detectors
                                                │
                                                ▼
                                         Isolation Forest
                                         and other models
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                          Fusion Layer
                           (Future)
                               │
                               ▼
                     Final Anomaly Decision
```

The main philosophy is:

Different detectors are good at detecting different types of anomalies. The final system will combine their evidence rather than expecting any individual detector to solve every anomaly detection problem.

---

## 2. Quick Start

This section describes the recommended path from cloning the repository to running and developing the project.

### Clone the repository

Clone the repository and switch to the ML engine branch:

```bash
git clone -b dev/ml-engine https://github.com/Root3141/SIH_2026.git
cd SIH_2026
```

If you have already cloned the repository:

```bash
git fetch origin
git checkout dev/ml-engine
git pull origin dev/ml-engine
```

### Create and activate a virtual environment

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

### Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, if using the project configuration:

```bash
pip install -e .
```

### Check the data

The project expects raw datasets under:

```text
data/raw/
```

Expected files include:

```text
ncr_weather_historical.parquet
ncr_weather_2026_present.parquet
```

CSV versions may also be present.

If the required raw datasets are missing or need to be regenerated, inspect and run:

```text
src/skyguard/data/fetch_data.py
```

This module is responsible for fetching and preparing the raw weather datasets used by the project.

### Verify the project

From the project root, try running:

```bash
python src/skyguard/evaluation/evaluate_spatial.py
```

This should:

* Load historical and evaluation datasets.
* Calibrate the spatial detector.
* Inject synthetic anomalies into evaluation data.
* Run the detector.
* Calculate performance metrics.
* Save results under `results/spatial/`.

You can also run the statistical detector evaluation:

```bash
python src/skyguard/evaluation/evaluate_statistical.py
```

Results will be saved under:

```text
results/statistical/
```

### Ready to work

Once the evaluation scripts run successfully, the development environment is ready.

The main locations to work with are:

```text
src/skyguard/detectors/     Detection algorithms
src/skyguard/evaluation/    Detector evaluation pipelines
src/skyguard/simulation/    Synthetic anomaly generation
src/skyguard/data/          Data fetching and preparation
src/skyguard/config/        Shared configuration and paths
```

Before modifying or adding a detector, read the sections describing the project architecture, data separation, and recommended detector workflow.
---

## 3. Project Structure

Current repository structure:

```text
skyguard/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│
├── results/
│   ├── statistical/
│   └── spatial/
│
└── src/
    └── skyguard/
        ├── config/
        ├── data/
        ├── detectors/
        ├── evaluation/
        ├── fusion/
        └── simulation/
```

### Important directories

#### `data/`

Contains project datasets.

```text
data/
├── raw/
│   ├── ncr_weather_historical.parquet
│   ├── ncr_weather_historical.csv
│   ├── ncr_weather_2026_present.parquet
│   └── ncr_weather_2026_present.csv
│
├── processed/
│
└── synthetic/
    └── spatial_evaluation_injected.parquet
```

#### `src/skyguard/detectors/`

Contains anomaly detection algorithms.

Current detectors:

```text
detectors/
├── statistical.py
└── spatial.py
```

Future detectors should also be added here. For example:

```text
detectors/
├── statistical.py
├── spatial.py
└── isolation_forest.py
```

#### `src/skyguard/evaluation/`

Contains scripts for evaluating detectors.

```text
evaluation/
├── evaluate_statistical.py
└── evaluate_spatial.py
```

Each detector should ideally have its own evaluation script.

The evaluation pipeline generally follows:

```text
Historical Data
      │
      ▼
Calibrate / Train Detector
      │
      ▼
Unseen Evaluation Data
      │
      ▼
Inject Synthetic Anomalies
      │
      ▼
Run Detector
      │
      ▼
Compare Predictions with Ground Truth
      │
      ▼
Calculate Metrics
      │
      ▼
Save Results
```

#### `src/skyguard/simulation/`

Contains tools for generating synthetic data.

Currently:

```text
simulation/
└── anomaly_injector.py
```

This module is used to inject controlled synthetic anomalies into otherwise clean evaluation data.

This is important because real-world anomaly ground truth is limited or unavailable.

#### `src/skyguard/config/`

Contains project-wide configuration.

Currently:

```text
config/
└── paths.py
```

Use this area for shared paths and configuration instead of hardcoding paths throughout the project.

#### `src/skyguard/fusion/`

Reserved for the future detector fusion system.

Currently:

```text
fusion/
└── __init__.py
```

Eventually this will combine evidence from:

- Statistical detector
- Spatial detector
- Isolation Forest
- Other ML models

#### `results/`

Contains outputs generated by detector evaluations.

Results are separated by detector:

```text
results/
├── statistical/
└── spatial/
```

Generated files should not be treated as source code. They are evaluation artifacts that can be regenerated by running the corresponding evaluation scripts.

---

## 4. Data

The project currently uses two main time periods.

### Historical data

File:

```text
data/raw/ncr_weather_historical.parquet
```

Period:

```text
2023-01-01 to 2025-12-31
```

Purpose:

Calibration and training.

This dataset is used to establish what normal behavior looks like.

Examples:

- Statistical thresholds
- Spatial residual distributions
- Future ML model training

### Unseen evaluation data

File:

```text
data/raw/ncr_weather_2026_present.parquet
```

Period:

```text
2026-01-01 to present
```

Purpose:

Detector evaluation.

This data should remain separate from historical calibration data.

The intended workflow is:

```text
Historical Data
      │
      └──► Calibration / Training

2026 Evaluation Data
      │
      └──► Synthetic Anomaly Injection
                  │
                  ▼
             Detector Evaluation
```

This separation helps prevent evaluation leakage.

---

## 5. Synthetic Anomaly Injection

Location:

```text
src/skyguard/simulation/anomaly_injector.py
```

The anomaly injector creates controlled faults in clean weather data.

It is used primarily for evaluating whether detectors can detect known anomalies.

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

The injector returns:

- `corrupted_df`: the original dataset with synthetic anomalies injected
- `injection_log`: a record of anomaly events that were created

### Supported anomaly types

Currently the injector supports:

| Anomaly | Description |
| --- | --- |
| spike | Sudden extreme deviation |
| offset | Sustained shift from normal value |
| drift | Gradual movement away from normal |
| stuck | Sensor value becomes constant |
| rate_change | Abnormally rapid change/ramp |
| noise | Increased random variation |
| dropout | Missing/invalid readings |

Not every detector is expected to detect every anomaly type equally well.

Examples:

- Spatial detection should be useful for isolated offsets.
- Temporal/statistical methods may be better for stuck sensors.
- Missing-value checks are naturally better suited to dropouts.

This is intentional. The project uses multiple complementary detectors.

### Ground truth

Injected anomalies include metadata that allows predictions to be compared with known ground truth.

Evaluation scripts may map injector metadata to standard columns such as:

- `is_anomaly`
- `anomaly_id`
- `anomaly_type`
- `anomaly_variable`
- `anomaly_severity`

These columns are used during evaluation.

---

## 6. Statistical Detector

Location:

```text
src/skyguard/detectors/statistical.py
```

The statistical detector identifies observations that are unusual based on historical and temporal behavior.

It is primarily intended to detect:

- Extreme values
- Sudden changes
- Temporal deviations
- Other statistically unusual behavior

The statistical detector should be evaluated using:

```bash
python src/skyguard/evaluation/evaluate_statistical.py
```

Results are saved under:

```text
results/statistical/
```

Current outputs include files such as:

- `alerts.csv`
- `alert_summary.csv`
- `station_statistics.csv`
- `variable_involvement.csv`
- `evidence_combinations.csv`
- `full_results.parquet`
- `detector_evaluation.png`

The exact outputs may evolve as the evaluation pipeline is improved.

---

## 7. Spatial Detector

Location:

```text
src/skyguard/detectors/spatial.py
```

The spatial detector compares each station's observation with an expected value estimated from nearby stations.

It uses spatial relationships between stations rather than only looking at the station's historical behavior.

Conceptually:

```text
Station A
Observed value: 35°C

Nearby stations:
B: 22°C
C: 23°C
D: 21°C

Spatial expectation ≈ 22°C

Residual:
35 - 22 = 13°C

Large residual → potentially anomalous
```

The expected value is calculated using Inverse Distance Weighting (IDW).

Current configuration testing settled on:

- IDW power: `2.0`
- Anomaly threshold: approximately `P99.5` calibration threshold

These settings should not be changed casually without evaluating the effect.

### Spatial detector strengths

The spatial detector is particularly useful for identifying:

- Isolated sensor offsets
- Localized spikes
- A station behaving inconsistently with nearby stations
- Spatially isolated faults

### Spatial detector limitations

The spatial detector is not expected to be the best detector for:

- Dropouts
- Stuck sensors
- Slow temporal drift
- Some gradual rate changes
- Region-wide weather events

For example, if every station experiences the same genuine weather event, that should generally not be classified as an anomaly.

The spatial detector includes evaluation for this behavior through a regional event sanity test.

### Run spatial evaluation

From the project root:

```bash
python src/skyguard/evaluation/evaluate_spatial.py
```

The script performs:

1. Load historical data
2. Load unseen evaluation data
3. Calibrate spatial detector
4. Inject synthetic anomalies
5. Run spatial detector
6. Calculate observation-level metrics
7. Calculate event-level metrics
8. Analyze false positives
9. Run regional event sanity test
10. Save results

Results are saved to:

```text
results/spatial/
```

---

## 8. Understanding Evaluation

The project evaluates detectors against synthetic anomalies because the exact ground truth of real-world anomalies is generally unavailable.

The standard evaluation workflow is:

```text
                    Historical Dataset
                           │
                           ▼
                    Calibrate Detector
                           │
                           ▼
                  Unseen Clean Dataset
                           │
                           ▼
                 Inject Known Anomalies
                           │
                           ▼
                    Corrupted Dataset
                           │
                           ▼
                     Run Detector
                           │
                           ▼
              Compare with Ground Truth
                           │
                           ▼
                    Performance Metrics
```

### Observation-level metrics

Each individual row or observation is evaluated.

Important metrics:

- True Positive: an anomaly was injected and detected
- False Positive: the detector flagged an observation that was not injected as anomalous
- False Negative: an anomaly was injected but not detected
- True Negative: a normal observation was correctly left unflagged

#### Precision

```text
TP / (TP + FP)
```

Of everything flagged by the detector, how much was actually anomalous?

#### Recall

```text
TP / (TP + FN)
```

Of all injected anomalies, how many were detected?

#### F1 Score

```text
2 × Precision × Recall
──────────────────────
   Precision + Recall
```

Balances precision and recall.

### Event-level metrics

An anomaly event may span multiple observations.

For example:

- Hour 1 → anomaly
- Hour 2 → anomaly
- Hour 3 → anomaly
- Hour 4 → anomaly

Instead of requiring every observation to be detected, event-level evaluation asks:

> Was this anomaly event detected at least once?

This is useful because operationally detecting an anomaly event may matter more than flagging every individual affected row.

---

## 9. Spatial Evaluation Results

Running:

```bash
python src/skyguard/evaluation/evaluate_spatial.py
```

produces files such as:

| File | Purpose |
| --- | --- |
| `observation_metrics.csv` | Overall precision, recall, F1, etc. |
| `event_metrics.csv` | Event-level detection performance |
| `performance_by_anomaly_type.csv` | Performance for each anomaly type |
| `performance_by_variable.csv` | Performance for temperature, humidity, pressure |
| `false_positive_analysis.csv` | False positive breakdown |
| `severity_distribution.csv` | Spatial severity distribution |
| `regional_event_test.csv` | Regional weather event sanity test |
| `injection_log.csv` | Synthetic anomaly events injected |
| `anomaly_predictions.csv` | Detector predictions |
| `full_results.parquet` | Complete evaluation dataset |
| `spatial_detector_evaluation.png` | Evaluation visualization |

The injected dataset may also be saved under:

```text
data/synthetic/
```

---

## 10. Recommended Workflow When Working on a Detector

When modifying an existing detector or adding a new one, use this workflow.

### Step 1: Understand the detector's purpose

Do not expect every detector to identify every anomaly.

Ask:

> What type of anomaly signal is this detector specifically designed to detect?

Examples:

| Detector | Primary signal |
| --- | --- |
| Statistical | Temporal/statistical deviation |
| Spatial | Disagreement with neighboring stations |
| Isolation Forest | Multivariate unusual patterns |

### Step 2: Calibrate or train using historical data

Use:

```text
2023–2025 historical data
```

Avoid using the unseen evaluation period for fitting thresholds or training models.

### Step 3: Evaluate on unseen data

Use:

```text
2026-present data
```

Inject synthetic anomalies using:

```text
src/skyguard/simulation/anomaly_injector.py
```

### Step 4: Evaluate performance

At minimum, inspect:

- Precision
- Recall
- F1 score
- False positive rate
- Performance by anomaly type
- Performance by variable
- Event-level recall

Do not optimize only one metric.

For example, reducing false positives by making thresholds extremely strict may destroy recall.

### Step 5: Test meaningful configuration changes

Avoid random parameter tuning.

Make a hypothesis first.

Example:

> Hypothesis: the anomaly threshold is too sensitive.

Change:

```text
P99 → P99.5
```

Measure:

- False positives
- Precision
- Recall
- Event-level recall

Keep changes that provide a meaningful tradeoff.

---

## 11. Adding a New Detector

New detectors should generally follow this structure.

### Create the detector

Add:

```text
src/skyguard/detectors/<detector_name>.py
```

The detector should ideally contain:

- Configuration
- Calibration / Training
- Detection
- Output formatting
- Self-test

### Create an evaluation script

Add:

```text
src/skyguard/evaluation/evaluate_<detector_name>.py
```

The evaluation script should follow the existing pattern:

```text
Load historical data
      │
      ▼
Train / Calibrate
      │
      ▼
Load evaluation data
      │
      ▼
Inject synthetic anomalies
      │
      ▼
Run detector
      │
      ▼
Evaluate predictions
      │
      ▼
Save results
```

Results should preferably go to:

```text
results/<detector_name>/
```

### Keep output conventions consistent

Where practical, detectors should produce compatible concepts:

- `alert`
- `severity`
- detector-specific evidence

Evaluation data should use standard ground truth concepts:

- `is_anomaly`
- `anomaly_id`
- `anomaly_type`
- `anomaly_variable`
- `anomaly_severity`

Consistent outputs will make the future fusion layer much easier to implement.

---

## 12. Important Project Principles

### Detectors are complementary

Do not judge every detector solely by whether it detects every anomaly type.

Examples:

- A spatial detector should not necessarily be expected to detect a dropout.
- A temporal detector may miss a spatially isolated offset.
- An ML detector may detect complex patterns missed by simple rules.

The final system is intended to combine multiple sources of evidence.

### Avoid data leakage

Keep calibration/training data separate from evaluation data.

Current intended split:

```text
Historical:
2023–2025
→ Calibration / Training

Evaluation:
2026-present
→ Testing
```

Synthetic anomalies provide ground truth.

Real anomaly labels are limited.

The anomaly injector provides controlled anomalies with known:

- Type
- Location
- Variable
- Severity
- Event identity

This allows objective evaluation.

### Don't overfit to synthetic evaluation

Synthetic anomalies are useful for benchmarking, but they are still simulated.

Avoid tuning a detector excessively to one exact injection configuration.

When changing detector logic, consider whether the improvement represents a genuinely better detection method or merely better alignment with the current synthetic anomalies.

---

## 13. Current Project Status

Currently implemented:

- ✅ Data collection and validation
- ✅ Historical/evaluation data separation
- ✅ Statistical detector
- ✅ Spatial detector
- ✅ Synthetic anomaly injector
- ✅ Spatial detector evaluation pipeline
- ✅ Spatial detector diagnostics
- ✅ Regional event sanity testing

Current detector architecture:

```text
                    ┌─────────────────┐
                    │ Weather Data    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      Statistical Detector           Spatial Detector
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                       Future Fusion
```

---

## 14. What’s Next?

The next major component is:

```text
Isolation Forest
```

The Isolation Forest model will add a multivariate anomaly detection approach.

Unlike the current detectors:

- Statistical detection focuses on statistical/temporal deviations.
- Spatial detection focuses on disagreement with neighboring stations.
- Isolation Forest can identify unusual combinations and multivariate patterns.

The intended workflow will be:

1. Decide features
2. Train on historical normal behavior
3. Run on unseen 2026 data
4. Evaluate using synthetic anomaly injection
5. Compare performance with existing detectors
6. Save results under `results/isolation_forest/`

After additional detectors are implemented, the next major task will be:

### Detector Fusion

The fusion layer will combine evidence from multiple detectors.

Conceptually:

```text
Statistical Alert ──────┐
                        │
Spatial Alert ──────────┼──► Fusion ──► Final Decision
                        │
Isolation Forest ───────┤
                        │
Future Detectors ───────┘
```

The goal is to produce a more reliable final anomaly decision than any individual detector alone.

---

## 15. Quick Command Reference

From the project root:

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run spatial evaluation

```bash
python src/skyguard/evaluation/evaluate_spatial.py
```

### Run statistical evaluation

```bash
python src/skyguard/evaluation/evaluate_statistical.py
```

### Important source locations

- Statistical detector: `src/skyguard/detectors/statistical.py`
- Spatial detector: `src/skyguard/detectors/spatial.py`
- Anomaly injector: `src/skyguard/simulation/anomaly_injector.py`
- Spatial evaluation: `src/skyguard/evaluation/evaluate_spatial.py`
- Statistical evaluation: `src/skyguard/evaluation/evaluate_statistical.py`
- Project paths: `src/skyguard/config/paths.py`

---

## Final Note

This project is evolving quickly.

When adding a major component:

- Keep the existing project structure consistent.
- Reuse the anomaly injection system for evaluation.
- Keep calibration/training separate from evaluation.
- Save outputs under `results/<detector_name>/`.
- Add a basic self-test where practical.
- Document important design decisions.
- Avoid optimizing a detector in isolation when the final system will use detector fusion.

The goal is not to make every individual detector perfect.

The goal is to build multiple detectors that provide different, useful signals which can later be combined into a stronger anomaly detection system.