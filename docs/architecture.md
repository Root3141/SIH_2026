# System Architecture

## High-level flow

```text
Historical Weather Data
        |
        v
Calibration / Training
        |
        v
   Frozen Detector
      Artifacts
        |
        +-------------------+
        |                   |
        v                   v
2026 Evaluation Data    Streamlit App
        |                   ^
        v                   |
  Detector Pipeline         |
        |                   |
   +----+----+----+         |
   |         |    |         |
   v         v    v         |
Statistical Spatial LSTM    |
 Detector   Detector Autoencoder
   |         |    |         |
   +---------+----+---------+
             |
             v
       Fusion Layer
         (2-of-3)
             |
             v
     Anomaly Decision
             |
             v
 Evidence / Explainability
             |
             v
        Diagnosis
             |
             +-------------> Dashboard
```

## Components

### Data and Calibration

Historical weather station observations are used to establish normal behaviour and calibrate the detectors.

The system keeps historical calibration data separate from the unseen evaluation period to avoid evaluation leakage.

### Statistical Detector

Detects anomalies using temporal and historical statistical signals.

It considers:

* Range violations
* Rate-of-change anomalies
* Rolling z-score deviations
* Persistence behaviour
* Historical baseline deviations

### Spatial Detector

Compares each weather station with nearby stations at the same timestamp.

It uses neighbouring stations and Inverse Distance Weighting (IDW) to estimate an expected value. Large differences between the observed and expected values indicate possible spatial anomalies.

### LSTM Autoencoder

Models multivariate temporal behaviour across:

* Temperature
* Atmospheric pressure
* Relative humidity

The detector reconstructs temporal windows and uses reconstruction error as an anomaly signal.

### Fusion Layer

The three detector outputs are combined using the current 2-of-3 decision policy.

An anomaly alert is raised when at least two detectors agree.

The fusion layer is deterministic; explainability does not change detector votes, thresholds, or the final decision.

### Explainability and Diagnosis

For detected alerts, SkyGuard can provide supporting information from each detector, including statistical checks, spatial comparisons, and reconstruction evidence.

Optional model-level explanations can show which observations contributed most to the reconstruction anomaly score.

The diagnosis layer provides operational hypotheses and recommended checks. These are treated as evidence-based hypotheses rather than confirmed physical failure modes.

### Streamlit Dashboard

The frontend presents the detector results to the user through the Streamlit application.

It provides:

* Weather station monitoring
* Anomaly alerts
* Detector evidence
* Station-level details
* Explainability information
* Operational diagnosis

The deployed prototype is available at:

`https://skyguardai-sih.streamlit.app/`

## Runtime flow

The deployed prototype uses pre-prepared detector artifacts and evaluation results.

```text
Prepared Detector Artifacts
          |
          v
   Frozen Detector State
          |
          v
    Fusion / Results
          |
          v
   Streamlit Dashboard
```

The dashboard does not retrain models or recalibrate detectors when it starts.

## Design principles

### Multiple complementary detectors

SkyGuard uses different detectors because different anomaly types have different signatures. Statistical, spatial, and temporal modelling provide complementary sources of evidence.

### Chronological separation

Historical data is used for calibration and training, while later unseen data is used for evaluation.

### Deterministic fusion

The final anomaly decision follows the configured detector-vote policy rather than an explanation model.

### Evidence-based explanations

Explainability exposes the evidence already used by the system and, where available, provides additional model-level attribution. It does not alter the underlying anomaly decision.
