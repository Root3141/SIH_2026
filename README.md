# SIH 2026 Project Repository

This repository contains **SkyGuard AI**, an AI/ML-based intelligent anomaly detection system for Automatic Weather Stations (AWS), developed for **Smart India Hackathon 2026 — Problem #73**.

## 1. Project Information

* **Project Title:** SkyGuard AI – Intelligent Real-Time Anomaly Detection for Automatic Weather Stations
* **PS ID:** SIH26073
* **PS Title:** AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)
* **Category:** Software
* **Theme:** Disaster Management
* **Department:** India Meteorological Department (IMD)
* **Organization:** Ministry of Earth Sciences (MoES)

## 2. Problem Statement

Automatic Weather Stations (AWS) continuously collect atmospheric observations that support weather forecasting, climate monitoring, disaster management, aviation, agriculture, and scientific research.

However, AWS observations can contain anomalies caused by sensor malfunction, communication failures, calibration drift, power fluctuations, harsh environmental conditions, and data corruption.

Traditional threshold-based quality-control methods may be insufficient for detecting complex, multivariate, or hidden anomalies. Erroneous observations can therefore propagate into downstream forecasting and decision-making systems.

The SIH problem requires an intelligent system capable of identifying abnormal, inconsistent, or faulty observations from AWS data using:

* Temperature (°C)
* Atmospheric Pressure (hPa)
* Relative Humidity (%)

The system should distinguish genuine meteorological events from sensor and data anomalies while minimizing false alarms and supporting scalable deployment across weather observation networks.

## 3. Proposed Solution

**SkyGuard AI** is a multi-detector anomaly detection framework designed to identify abnormal AWS observations using temporal, statistical, spatial, and multivariate behavioural patterns.

SkyGuard combines three complementary detection approaches:

1. **Statistical Detector**
   Detects unusual observations using range checks, rate-of-change behaviour, statistical deviations, persistence signals, and historical patterns.

2. **Spatial Detector**
   Compares a station's observations with nearby stations using spatial relationships and Inverse Distance Weighting (IDW). This helps identify observations that are inconsistent with surrounding stations.

3. **LSTM Autoencoder**
   Learns normal multivariate temporal behaviour across temperature, pressure, and humidity. Anomalies are identified through elevated reconstruction error.

The detector outputs can then be combined through a **fusion layer**, allowing multiple sources of evidence to contribute to the final anomaly decision.

SkyGuard is evaluated using controlled synthetic anomaly injection with known ground truth. This enables systematic measurement of detection accuracy, false-positive behaviour, event-level detection, and detector complementarity.

## 4. Key Features

* Real-time-oriented AWS anomaly detection architecture
* Multivariate analysis of temperature, pressure, and humidity
* Statistical anomaly detection
* Spatial consistency analysis across nearby weather stations
* LSTM Autoencoder for temporal anomaly detection
* Detection of spikes, offsets, drift, stuck values, rapid rate changes, noise, and other anomalous behaviours
* Detector fusion for combining complementary evidence
* Observation-level anomaly scoring
* Event-level anomaly detection and evaluation
* Severity and confidence-oriented anomaly outputs
* Explainable anomaly contributors
* Station-level anomaly visualization
* Interactive Streamlit dashboard
* Historical and synthetic anomaly evaluation framework
* Frozen calibration state for reproducible inference
* Support for scalable deployment across multiple AWS stations

## 5. Technology Stack

* **Frontend / Dashboard:** Streamlit
* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy, PyArrow
* **Machine Learning:** Scikit-learn, TensorFlow/Keras
* **Deep Learning:** LSTM Autoencoder
* **Explainability:** SHAP, LIME
* **Spatial Analysis:** Inverse Distance Weighting (IDW)
* **Data Format:** CSV, Parquet
* **Visualization:** Plotly, Matplotlib
* **Model Artifacts:** Pickle
* **Deployment:** Streamlit Community Cloud / Cloud deployment

## 6. Architecture

SkyGuard uses multiple complementary anomaly detectors operating on AWS observations.

```text
                    AWS Observations
                           │
                           ▼
             Temperature / Pressure / Humidity
                           │
                           ▼
                  Data Processing Layer
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       Statistical      Spatial         LSTM
        Detector       Detector      Autoencoder
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                    Fusion Engine
                           │
                           ▼
              Anomaly Score / Alert
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          Severity      Explanation   Sensor Health
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  Streamlit Dashboard
```

The system is designed around complementary evidence rather than relying on a single anomaly detector.

## 7. Repository Structure

```text
SKYGUARD/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── artifacts/
│   └── skyguard_v1/
│       ├── fusion_config.json
│       ├── lstm.pkl
│       ├── spatial.pkl
│       └── statistical.pkl
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── docs/
│   ├── CODEBASE_REFERENCE.md
│   ├── DEVELOPER_GUIDE.md
│   └── LAST_STEPS.md
│
├── frontend/
│   └── streamlit/
│       ├── app.py
│       ├── pages/
│       │   └── 1_Station_Detail.py
│       └── utils/
│           ├── detector.py
│           ├── simulator.py
│           └── styles.py
│
├── notebooks/
│   ├── data_validation.ipynb
│   └── lstm_autoencoder_exploration.ipynb
│
├── results/
│   ├── fusion/
│   ├── lstm_autoencoder/
│   ├── prototype/
│   ├── spatial/
│   ├── statistical/
│   ├── statistical_spatial/
│   └── statistical_spatial_lstm/
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

### What goes where?

| Item                          | Location                   |
| ----------------------------- | -------------------------- |
| Core source code              | `src/skyguard/`            |
| Streamlit dashboard           | `frontend/streamlit/`      |
| Detector implementations      | `src/skyguard/detectors/`  |
| Fusion engine                 | `src/skyguard/fusion/`     |
| Evaluation pipelines          | `src/skyguard/evaluation/` |
| Synthetic anomaly generation  | `src/skyguard/simulation/` |
| Model/configuration artifacts | `artifacts/`               |
| Raw and synthetic datasets    | `data/`                    |
| Evaluation results            | `results/`                 |
| Technical documentation       | `docs/`                    |
| Exploratory notebooks         | `notebooks/`               |
| Project overview              | `README.md`                |

## 8. Detection Approach

### Statistical Detector

The statistical detector identifies observations that deviate from expected temporal and historical behaviour.

It considers signals including:

* Range violations
* Rate-of-change anomalies
* Z-score deviations
* Persistence behaviour
* Historical deviations

### Spatial Detector

The spatial detector evaluates whether a station is consistent with nearby stations.

For example:

```text
Station A: 35°C

Nearby stations:
B: 22°C
C: 23°C
D: 21°C

Expected spatial value ≈ 22°C

Residual ≈ 35 - 22 = 13°C

Large residual → Possible anomaly
```

The evaluated configuration uses:

```text
Neighbors:              4
IDW power:              2.0
Suspicious percentile:  95
Anomaly percentile:     99.5
```

### LSTM Autoencoder

The LSTM Autoencoder learns normal multivariate temporal behaviour across:

```text
Temperature
Pressure
Humidity
```

Normal sequences are reconstructed with relatively low error, while unusual sequences tend to produce larger reconstruction errors.

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
            │
            ▼
    Reconstruction Error
            │
            ▼
      Anomaly Score
```

The LSTM respects station boundaries, avoids bridging large temporal gaps, and follows chronological training and calibration.

## 9. Anomaly Types

SkyGuard's synthetic evaluation framework supports controlled injection of several anomaly types:

| Anomaly Type | Description                                 |
| ------------ | ------------------------------------------- |
| Spike        | Sudden extreme deviation                    |
| Offset       | Sustained shift from expected values        |
| Drift        | Gradual movement away from normal behaviour |
| Stuck        | Sensor becomes constant                     |
| Rate Change  | Abnormally rapid temporal change            |
| Noise        | Increased random variation                  |
| Dropout      | Missing or invalid readings                 |

Synthetic anomaly labels are used for **offline evaluation only** and are not used to train or recalibrate the production detector during normal inference.

## 10. Evaluation

SkyGuard uses a controlled synthetic benchmark to evaluate detector performance.

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
Run Detector
        │
        ▼
Compare with Ground Truth
        │
        ▼
Calculate Performance
```

Evaluation includes both observation-level and event-level metrics.

### Observation-Level Metrics

* Precision
* Recall
* F1 Score
* False Positive Rate
* True Positives
* False Positives
* False Negatives
* True Negatives

### Event-Level Evaluation

An anomaly event is considered detected when at least one anomalous observation within the event is flagged.

This provides an operational perspective in which detecting an anomaly event early can be more important than detecting every individual anomalous observation.

### Detector Complementarity

The system also evaluates whether the LSTM provides useful detections that are missed by the statistical and spatial detectors.

This supports the design of the final fusion strategy.

## 11. Installation

Clone the repository and create a Python environment:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 12. Run

### Streamlit Dashboard

From the repository root:

```bash
streamlit run frontend/streamlit/app.py
```

The dashboard provides an interactive interface for exploring SkyGuard's anomaly detection results and station-level behaviour.

### Evaluation

The individual detector evaluation pipelines can be executed with:

```bash
python src/skyguard/evaluation/evaluate_statistical.py
```

```bash
python src/skyguard/evaluation/evaluate_spatial.py
```

The LSTM Autoencoder supports:

```bash
python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode full
```

```bash
python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode score
```

```bash
python src/skyguard/evaluation/evaluate_lstm_autoencoder.py --mode analyze
```

The combined detector evaluation can be run using:

```bash
python src/skyguard/evaluation/evaluate_combined.py
```

Evaluation outputs are stored under the corresponding directories in `results/`.

## 13. Deployment

SkyGuard's Streamlit dashboard is designed to support simple cloud deployment.

For the prototype/demo deployment, the dashboard can load precomputed inference results rather than retraining models or rerunning the complete evaluation pipeline at application startup.

Recommended deployment flow:

```text
GitHub Repository
        │
        ▼
Streamlit Community Cloud
        │
        ▼
frontend/streamlit/app.py
        │
        ▼
SkyGuard Dashboard
```

The Streamlit entrypoint is:

```text
frontend/streamlit/app.py
```

## 14. Expected Outputs

SkyGuard is designed to provide:

* Real-time-oriented anomaly alerts
* Anomaly scores
* Severity indicators
* Detector-specific evidence
* Root-cause/anomaly-type information where available
* Station-level anomaly visualization
* Temporal anomaly analysis
* Spatial consistency analysis
* Sensor health-oriented information
* Explainable anomaly contributors
* Corrected or estimated values as a potential future extension

## 15. Example Use Case

Consider an Automatic Weather Station that suddenly reports:

```text
Temperature: 55°C
Humidity:    Extremely high
Pressure:    Abnormal variation
```

while nearby stations continue reporting normal observations.

SkyGuard can combine:

* Temporal/statistical evidence
* Spatial inconsistency with neighbouring stations
* Multivariate temporal reconstruction error

to determine whether the observation is more likely to represent a genuine meteorological event or a sensor/data anomaly.

The resulting evidence can be surfaced through the dashboard for further investigation.

## 16. Future Scope

Potential extensions include:

* Live AWS data-stream integration
* Direct integration with meteorological observation networks
* Dedicated missing-data and communication-error detection
* Sensor degradation forecasting
* Predictive maintenance recommendations
* Automatic anomalous-value correction/imputation
* Edge AI deployment on low-power hardware such as ESP32-class devices
* More advanced learned fusion/meta-models
* Adaptive station-specific calibration
* Large-scale deployment across national AWS networks
* Continuous model monitoring and recalibration workflows
* Integration with operational weather-quality-control systems

## 17. Documentation

Additional technical documentation is available under `docs/`:

* `docs/CODEBASE_REFERENCE.md` — Codebase and detector reference
* `docs/DEVELOPER_GUIDE.md` — Development and evaluation workflow
* `docs/LAST_STEPS.md` — Current project status and finalization notes

## 18. Demo Video

A demo video is recommended for the final SIH submission.

Add the YouTube or Google Drive link here:

```text
<DEMO_VIDEO_URL>
```

## 19. Screenshots / Prototype

Important dashboard screenshots, system demonstrations, and prototype images should be stored under:

```text
assets/screenshots/
```

Recommended examples include:

```text
assets/screenshots/
├── dashboard-overview.png
├── station-detail.png
├── anomaly-alert.png
└── detector-comparison.png
```

## 20. Final Presentation

The final SIH presentation should be maintained in the repository whenever file size permits.

Recommended location:

```text
submission/
└── PRESENTATION.md
```

If the presentation file is too large for GitHub, provide an accessible Google Drive or OneDrive viewer link in `submission/PRESENTATION.md`.

## Important

Before submission, ensure that the repository is accessible to SIH reviewers.

**Do not upload:**

* Passwords
* API keys
* Access tokens
* `.env` files containing secrets
* Private credentials
* Other confidential information

All datasets, model artifacts, and generated results included in the repository should be checked for licensing, size, and distribution requirements before final submission.

---

**SkyGuard AI — Building trustworthy weather observations through intelligent anomaly detection.**
