# SkyGuard — Fusion, Integration & Deployment Guide

## 1. Objective

The goal is to turn the existing three-detector evaluation system into a single usable prototype:

```text
                 ONE-TIME OFFLINE SETUP
                         │
              Historical 2023–2025 data
                         │
                         ▼
                 Calibration / setup
                         │
                         ▼
                 Saved detector state
                         │
                         │
                 PROTOTYPE RUNTIME
                         │
             Final injected 2026 dataset
                         │
                         ▼
              ┌──────────────────────┐
              │   SkyGuard Fusion    │
              │                      │
              │ Statistical           │
              │ Spatial               │
              │ LSTM Autoencoder      │
              │          ↓            │
              │       2 of 3          │
              └──────────┬───────────┘
                         │
                         ▼
                  Final results
                         │
                         ▼
                    Streamlit
                         │
                         ▼
                    Dashboard
```

The existing project already follows the necessary separation: the three detectors are calibrated independently, the LSTM is frozen for subsequent evaluation, and the fusion layer consumes detector outputs rather than modifying detector internals.
The prototype will therefore focus on **packaging and integrating what already exists**, rather than redesigning the detectors.

---

# 2. Final Prototype Architecture

Use four logical components.

```text
src/skyguard/
│
├── simulation/
│   └── create_final_dataset.py
│
├── fusion/
│   ├── calibrate.py
│   └── fusion.py
│
└── detectors/
    ├── statistical.py
    ├── spatial.py
    └── lstm_autoencoder.py

frontend/
└── streamlit/
    └── app.py
```

And the generated artifacts:

```text
data/
└── synthetic/
    ├── skyguard_demo_2026.parquet
    └── skyguard_demo_2026_log.parquet

artifacts/
└── skyguard_v1/
    ├── statistical.pkl
    ├── spatial.pkl
    ├── lstm.pkl
    └── fusion_config.json

results/
└── prototype/
    └── skyguard_demo_2026_results.parquet
```

The exact filenames can be adjusted to match the repository's current path conventions.

---

# 3. Step 1 — Create the Final 2026 Demo Dataset

## Purpose

Create one reproducible 2026 dataset containing the synthetic anomalies that the final prototype will demonstrate.

The current combined evaluator already performs this operation:

```text
clean 2026 data
      ↓
fixed anomaly seed = 42
      ↓
anomaly injection
      ↓
injected dataset
      +
ground-truth information
```

The current evaluator uses a 2% anomaly rate and random seed 42 and writes the shared injected benchmark to the synthetic-data directory.

## New file

Create:

```text
src/skyguard/simulation/create_final_dataset.py
```

Its responsibility should be **only**:

1. Load `ncr_weather_2026_present.parquet`.
2. Apply the final `AnomalyConfig`.
3. Preserve the anomaly ground-truth columns.
4. Save the final injected dataframe.
5. Save the injection log.
6. Print a reproducibility summary.

Example outputs:

```text
data/synthetic/skyguard_demo_2026.parquet
data/synthetic/skyguard_demo_2026_log.parquet
```

The injected dataset is the **input to the prototype**.

The injection log is used for evaluation/explanation, not as detector input.

---

# 4. Step 2 — Define What Calibration Means

Calibration is a **one-time offline process**.

It must not execute whenever Streamlit starts.

The historical dataset remains:

```text
2023-01-01 → 2025-12-31
```

and the 2026 dataset remains unseen evaluation/demo data. The project documentation explicitly requires this chronological separation.

## Calibration responsibilities

Calibration should:

```text
Historical 2023–2025
        │
        ├── Statistical calibration
        │
        ├── Spatial calibration
        │
        └── LSTM frozen calibration
                  │
                  ▼
          Saved inference artifacts
```

Create:

```text
src/skyguard/fusion/calibrate.py
```

This file should **not implement new anomaly-detection logic**.

It should reuse the existing detector APIs.

The statistical detector already exposes `calibrate_statistical_detector(...)`; the spatial detector already exposes `calibrate_spatial_detector(...)`.
The LSTM calibration already exists in `results/lstm_autoencoder/calibration.pkl`, which contains the trained model state, normalization state, configuration and calibrated thresholds.

## Important rule

Do **not** recalibrate from the injected 2026 dataset.

The synthetic labels are for evaluation, not detector calibration.

---

# 5. Step 3 — Save the Calibration Artifacts

The purpose of calibration is to create a stable artifact that runtime inference can load.

Conceptually:

```text
artifacts/
└── skyguard_v1/
    ├── statistical.pkl
    ├── spatial.pkl
    ├── lstm.pkl
    └── fusion_config.json
```

However, do not blindly serialize every object.

Save only what `fusion.py` actually needs at runtime.

## Statistical

The statistical detector's calibration includes:

```text
roc_thresholds
hourly_baselines
```

and its runtime detector uses those values to compute its evidence and final `statistical_alert`.

## Spatial

The spatial calibration includes:

```text
neighbors
residual_thresholds
```

and runtime inference uses these to calculate expected values, residuals and spatial anomaly flags.

## LSTM

Use the already-frozen calibration state.

The project already treats this as the frozen baseline and does not retrain it during combined evaluation.

For the prototype, whether this artifact needs to be copied into the deployed application depends on whether runtime `fusion.py` is going to execute LSTM inference or consume precomputed LSTM outputs.

---

# 6. Step 4 — Define the Fusion Policy

The final prototype policy is:

```text
Statistical alert
       +
Spatial alert
       +
LSTM alert

At least 2 TRUE
       ↓
FINAL ALERT
```

Formally:

```python
votes = (
    statistical_alert.astype(int)
    + spatial_alert.astype(int)
    + lstm_ae_alert.astype(int)
)

final_alert = votes >= 2
```

The existing combined evaluator already implements this exact operation for its `any_two_of_three_*` variants.

The existing fusion package also defines the 2-of-3 policy as the default count-tier strategy.

## Do not add another ML model here

For the prototype, fusion is a deterministic evidence-combination rule.

Do not introduce the logistic meta-model into the final runtime unless the team explicitly decides that it is the final evaluated strategy.

`evaluate_fusion.py` currently experiments with multiple strategies, including the logistic meta-model, but that is an evaluation tool—not automatically the production policy.

---

# 7. Step 5 — Create the Main Fusion Interface

Create:

```text
src/skyguard/fusion/fusion.py
```

This is the **only ML-facing function Streamlit should need**.

Define one main interface:

```python
def run_fusion(df):
    ...
```

Conceptually:

```text
Input dataframe
      │
      ▼
Statistical detector
      │
Spatial detector
      │
LSTM detector
      │
      ▼
Detector outputs
      │
      ▼
2-of-3 rule
      │
      ▼
Final result dataframe
```

## Input

The input must contain at least:

```text
station_id
timestamp
temperature
pressure
humidity
```

This matches the detector interfaces in the current code.

## Output

The runtime result should retain the original weather fields and add a clean set of fusion fields:

```text
station_id
timestamp
temperature
pressure
humidity

statistical_alert
spatial_alert
lstm_ae_alert

detector_votes
final_alert
final_severity
final_confidence
```

Keep detector-specific columns with their existing prefixes; the developer guide explicitly requires this convention to avoid ambiguous columns.

---

# 8. Step 6 — Define Severity

The prototype should distinguish:

```text
NORMAL
SUSPICIOUS
ANOMALY
CRITICAL
```

where supported by the existing detector evidence.

The existing statistical detector already distinguishes `normal`, `suspicious`, `anomaly`, and `critical`, and produces a corresponding severity score.

The fusion layer should then define a simple final policy.

Example:

```text
0 detector votes:
    NORMAL

1 detector vote:
    SUSPICIOUS

2 detector votes:
    ANOMALY

3 detector votes:
    CRITICAL
```

This mapping should be treated as a **prototype presentation policy**, not as a newly evaluated ML claim.

If the existing evaluated fusion implementation already defines a more appropriate tiering scheme, use that instead.

---

# 9. Step 7 — Define Confidence

Do not pretend that "2 of 3" is a statistically calibrated probability.

For the prototype, confidence can simply represent detector agreement:

```text
0/3 → 0%
1/3 → 33%
2/3 → 67%
3/3 → 100%
```

or, preferably, present the UI value as:

```text
Detector agreement: 2 / 3
```

rather than calling it a probability.

This is clearer and more defensible.

---

# 10. Step 8 — Decide Between Precomputed and Runtime Inference

For the hackathon prototype, use this approach:

```text
              ONE-TIME
                  │
                  ▼
       run detectors on final
          injected 2026 data
                  │
                  ▼
         save final results
                  │
                  ▼
       skyguard_demo_2026_results.parquet
                  │
                  ▼
              Streamlit
```

This has a major advantage:

**the dashboard does not need to retrain or run the LSTM just to display the demo.**

The generated result file becomes the stable presentation dataset.

The existing evaluation pipeline already produces a `combined_full_results.parquet`, and `evaluate_fusion.py` consumes that file directly.

---

# 11. Step 9 — Generate the Final Prototype Results

After calibration and finalizing the fusion policy:

```bash
python -m skyguard.simulation.create_final_dataset
```

then:

```bash
python -m skyguard.fusion.fusion
```

or provide a dedicated preparation command that performs:

```text
load final injected 2026 data
        ↓
load calibration artifacts
        ↓
run detectors
        ↓
apply 2-of-3
        ↓
save prototype results
```

Save:

```text
results/prototype/skyguard_demo_2026_results.parquet
```

The important property is:

> **The dashboard does not alter this file.**

It only reads it.

---

# 12. Step 10 — Replace the Existing Streamlit Simulation

The current `app.py` is still using its own synthetic simulator:

```python
generate_all_datasets()
evaluate_station(...)
```

and then manually advances through the simulation.

That should be removed.

The new architecture should instead be:

```python
results = load_results()
```

Then select the current timestamp:

```python
current = results[
    results["timestamp"] == selected_timestamp
]
```

The frontend should not independently decide whether something is anomalous.

It should simply display:

```text
final_alert
final_severity
detector_votes
```

returned by the fusion pipeline.

---

# 13. Step 11 — Make the Streamlit Dashboard

The dashboard should have three major views.

## A. Network overview

Show:

```text
Normal stations
Suspicious stations
Active anomalies
Total monitored stations
```

The current app already has this general structure, so that layout can be retained while replacing its simulator backend.

## B. Map

For each station:

```text
Station
Status
Temperature
Pressure
Humidity
```

The current Streamlit app already has a station map and station cards, so these can be preserved and connected to real fusion results instead.

## C. Alert detail

When a station is selected:

```text
AWS-042

Temperature: 41.8 °C
Pressure: 1002 hPa
Humidity: 88 %

FINAL STATUS
🔴 ANOMALY

Detector agreement
✓ Statistical
✓ Spatial
✗ LSTM

2 / 3 detectors agree
```

Then show the detector-specific reasoning underneath.

---

# 14. Step 12 — Add the Anomaly Timeline

For the selected station, show:

```text
Time
  │
  │       ┌── anomaly
  │       │
──┼───────●──────────────
  │
  └──────────────────────
```

At minimum:

```text
temperature over time
pressure over time
humidity over time
final_alert over time
```

For the demo dataset, also show the injected anomaly marker when ground truth is being demonstrated.

Do **not** expose ground-truth labels as detector output in a real inference view.

---

# 15. Step 13 — Add Detector Agreement

This is one of the most valuable visualizations for your architecture.

For each alert:

```text
                    Detector Evidence

Statistical      ✅
Spatial          ✅
LSTM             ❌

                 2 / 3
              FINAL ALERT
```

This directly demonstrates why the system is using multiple complementary detectors.

The project's stated philosophy is explicitly to combine complementary evidence rather than force every detector to solve every anomaly type.

---

# 16. Step 14 — Add Sensor Health

Create a station health summary based on recent results.

For example:

```text
AWS-042

Health: POOR

Recent alerts: 4
Suspicious readings: 7
Last alert: 14:00
Dominant issue: Temperature
```

The exact health formula should be kept simple and clearly documented.

Do not claim that this is a validated maintenance-prediction model unless the project has actually evaluated one.

## The problem statement does call for sensor health status and optionally maintenance/degradation prediction, so health status is appropriate for the prototype.

# 17. Step 15 — Keep Evaluation and Demo Ground Truth Separate

The injected dataset contains:

```text
is_anomaly
anomaly_id
anomaly_type
anomaly_variable
anomaly_severity
```

because those are required for evaluation. The existing combined evaluator ensures those fields exist after injection.

But the detector itself must not use these columns for inference.

Use them only for:

```text
offline evaluation
demo annotations
performance reporting
```

not:

```text
threshold calculation
detector decisions
runtime prediction
```

This follows the existing project rules.

---

# 18. Step 16 — Deployment Layout

For the prototype, the deployment package can be kept very small.

```text
SIH_2026/
│
├── src/
│   └── skyguard/
│       ├── detectors/
│       ├── fusion/
│       │   ├── calibrate.py
│       │   └── fusion.py
│       └── simulation/
│           └── create_final_dataset.py
│
├── data/
│   └── synthetic/
│       ├── skyguard_demo_2026.parquet
│       └── skyguard_demo_2026_log.parquet
│
├── results/
│   └── prototype/
│       └── skyguard_demo_2026_results.parquet
│
├── frontend/
│   └── streamlit/
│       └── app.py
│
├── artifacts/
│   └── skyguard_v1/
│
├── requirements.txt
└── README.md
```

---

# 19. Step 17 — Startup Behavior

When Streamlit starts, it should **not**:

```text
train model
calibrate thresholds
inject anomalies
run evaluation
```

It should only:

```text
load configuration
        ↓
load final result dataset
        ↓
display dashboard
```

This makes startup deterministic and fast.

---

# 20. Step 18 — Optional Streaming Simulation

Your current frontend simulates a live stream by advancing a `time_index`.

We can preserve that behavior.

Instead of generating simulated datasets, do:

```text
full 2026 results
       ↓
current time index
       ↓
current station snapshot
       ↓
dashboard
```

So the demo still appears real-time:

```text
14:00
  ↓
14:01
  ↓
14:02
  ↓
14:03
```

but the data is now coming from the **actual evaluated SkyGuard detector pipeline**.

That is much stronger for a hackathon demonstration than the current frontend's independent `evaluate_station()` simulator.

---

# 21. Step 19 — Testing Before Deployment

Before deployment, run these checks.

## Data check

Confirm:

```text
2026 dataset loads
station IDs are valid
timestamps are valid
temperature/pressure/humidity exist
```

## Detector check

Confirm:

```text
statistical_alert exists
spatial_alert exists
lstm_ae_alert exists
```

The project already standardizes detector output names this way.

## Fusion check

Confirm:

```text
votes ∈ {0,1,2,3}

final_alert == (votes >= 2)
```

## Reproducibility check

Confirm:

```text
same dataset
same anomaly seed
same calibration
same fusion policy
```

The project explicitly requires shared benchmark, shared anomaly injection and shared random seed for detector/fusion comparisons.

## Dashboard check

Verify:

```text
station map
station detail
alert log
severity
detector agreement
historical timeline
```

---

# 22. Step 20 — Deployment

For the prototype deployment:

```bash
streamlit run frontend/streamlit/app.py
```

The application should read the packaged prototype results.

The deployed application should **not require the historical 2023–2025 dataset** if all required inference results have already been generated.

Likewise, it does not need to rerun calibration.

This is the main benefit of doing calibration and inference preparation offline.

---

# 23. What Happens When the Model Changes?

The workflow becomes:

```text
MODEL CHANGE
     ↓
re-run calibration
     ↓
regenerate final 2026 benchmark
     ↓
re-evaluate fusion
     ↓
accept/reject new version
     ↓
regenerate prototype result dataset
     ↓
deploy
```

The dashboard itself does not need modification unless the output schema changes.

---

# 24. Recommended Development Order

Implement in this exact order:

```text
1. create_final_dataset.py
             ↓
2. calibrate.py
             ↓
3. fusion.py
             ↓
4. generate prototype results
             ↓
5. replace Streamlit simulator
             ↓
6. build final dashboard
             ↓
7. test
             ↓
8. deploy
```

Do not start with Streamlit.

The most important dependency is:

```text
simulation → calibration → fusion → result dataset → frontend
```

---

# 25. Final Responsibilities

## `simulation/create_final_dataset.py`

Responsible for:

```text
creating the fixed 2026 demo benchmark
```

## `fusion/calibrate.py`

Responsible for:

```text
one-time calibration
saving reusable detector state
```

## `fusion/fusion.py`

Responsible for:

```text
loading detector state
running detectors
applying 2-of-3
returning final decisions
```

## Streamlit

Responsible for:

```text
visualization
navigation
filtering
station views
alerts
timelines
```

It should not contain detector logic.

---

# 26. Final Runtime Contract

The most important interface in the whole system should be:

```python
results = run_fusion(data)
```

Everything upstream can change internally.

Everything downstream only needs to know:

```text
What was measured?
What did each detector say?
How many agreed?
What is the final status?
Why?
```

That gives the project a clean boundary between ML and frontend.

---

# 27. Prototype Success Criteria

The prototype is complete when all of the following are true:

```text
✓ One reproducible 2026 injected dataset exists
✓ Calibration runs separately and only once per model version
✓ Detector state is frozen
✓ 2-of-3 is deterministic
✓ Fusion returns a stable result schema
✓ Streamlit consumes fusion results rather than its own simulator
✓ Dashboard can replay the 2026 dataset as a stream
✓ Alerts show detector agreement
✓ Station details show sensor readings and reasoning
✓ No training occurs during Streamlit startup
✓ Evaluation ground truth is not used to make runtime decisions
✓ The full process can be reproduced from documented commands
```

The architecture remains consistent with the existing project's intended development order: detector fusion is the current next stage, followed by integration and then frontend deployment.
