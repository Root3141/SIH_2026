# SkyGuard Detector Documentation

Complete documentation of the SkyGuard anomaly detection system, covering statistical, spatial, anomaly injection, and evaluation modules.

---

## Table of Contents

1. Statistical Anomaly Detection Module
2. Spatial Consistency Detector Module
3. Anomaly Injector Module
4. Statistical Detector Evaluation Module
5. Spatial Detector Evaluation Module

---

# Statistical Anomaly Detection Module

**File**: `statistical.py`

This file implements statistical methods to detect anomalies in weather station observations (temperature, humidity, pressure). Here's a detailed breakdown:

## **Core Components**

### **Configuration Classes**

- **`StatisticalConfig`**: Stores detection parameters like z-score thresholds, time windows (24h, 168h), and persistence tolerances
- **`StatisticalCalibration`**: Holds pre-computed baselines and thresholds from training data

### **Detection Methods**

1. **Range Anomalies** (`detect_range_anomalies`)
   - Checks if values fall outside normal bounds (e.g., temperature: -10°C to 55°C)
   - Returns boolean series marking out-of-range values

2. **Rate of Change (ROC)** (`detect_roc_anomalies`)
   - Detects sudden jumps between consecutive readings
   - Compares absolute change to hourly percentile thresholds from calibration

3. **Z-Score Level Detection** (`calculate_rolling_zscore`)
   - Calculates how many standard deviations a value deviates from rolling mean
   - Uses grouped rolling windows per station to detect sustained deviations
   - Shifts calculations backward to avoid lookahead bias

4. **Persistence Detection** (`detect_persistence`)
   - Flags periods with abnormally low variance
   - Detects stuck/unchanging sensor readings (potential malfunction)

### **Calibration Process**

`calibrate_statistical_detector` preprocesses training data to:

- Compute hourly baselines (mean values by hour and station)
- Calculate ROC thresholds (99th percentile of hour-specific changes)
- Creates station-specific reference profiles

### **Evidence Aggregation**

`build_evidence_families` groups anomaly flags into four "families":

- **Range**: Out-of-bounds readings
- **ROC**: Rapid changes
- **Level**: Statistical deviations (z-score)
- **Persistence**: Low variability

### **Severity Assignment**

`assign_statistical_severity` scores alerts as:

- **Normal** (0.0): No anomalies
- **Suspicious** (0.33): Single evidence family triggered
- **Anomaly** (0.67): Multiple families OR persistence flag
- **Critical** (1.0): Out-of-range values

## **Main Workflow**

`run_statistical_detector` orchestrates the full pipeline:

1. Validates input data
2. Applies all detection methods per variable
3. Aggregates evidence into families
4. Assigns severity scores and alert flags

## **Testing**

`_run_self_test()` generates synthetic weather data and injects anomalies to verify detection works correctly.

---

# Spatial Consistency Detector Module

**File**: `spatial.py`

This file detects anomalies by comparing each weather station to nearby stations at the same timestamp. It identifies readings that deviate significantly from spatially interpolated expected values.

## **Core Components**

### **Configuration Classes**

- **`SpatialConfig`**: Controls detection behavior
  - `k_neighbors`: Number of nearest stations to use (default 4)
  - `idw_power`: Inverse distance weighting exponent (default 2.0)
  - Separate percentile thresholds for humidity (more variable) vs. temperature/pressure
  - `min_neighbors_required`: Minimum valid neighbors to make a prediction (default 2)

- **`SpatialCalibration`**: Pre-computed station metadata
  - `neighbors`: Weighted list of k-nearest neighbors per station
  - `residual_thresholds`: P95 and P99 percentile thresholds per station/variable

### **Distance Calculation**

**`haversine_km`**: Computes great-circle distance between two lat/lon coordinates using the Haversine formula. Returns distance in kilometers.

### **Neighbor Network Building**

**`build_neighbor_weights`**:

1. For each station, calculates distances to all other stations
2. Selects k-nearest neighbors
3. Applies inverse distance weighting: `weight = 1 / distance^power`
4. Normalizes weights to sum to 1.0
5. Returns ordered list of (neighbor_id, normalized_weight) tuples

### **Spatial Interpolation**

**`calculate_idw_expected`**:

1. Creates pivot table: rows=timestamps, columns=station_ids, values=variable
2. For each station, computes Inverse Distance Weighting (IDW) interpolation:
   - Multiplies neighbor values by their normalized weights
   - Handles missing data by tracking valid neighbors per timestamp
   - Returns expected values using weighted average of available neighbors
3. Also tracks neighbor count for quality assessment
4. Returns two aligned Series: expected values and neighbor counts

### **Calibration Process**

**`calibrate_spatial_detector`**:

1. Builds neighbor network using station coordinates
2. For each variable and station:
   - Calculates IDW expected values
   - Computes absolute residuals: `|observed - expected|`
   - Filters to rows with minimum valid neighbors (≥2)
   - Calculates P95 and P99 percentiles of residuals
   - Humidity uses stricter thresholds (P99.0 vs P99.5 for other variables)
3. Returns calibration object with thresholds and neighbor network

### **Detection Logic**

**`run_spatial_detector`**:

1. For each variable:
   - Calculates IDW expected values and residuals
   - Maps station→(P95, P99) thresholds from calibration
   - Flags suspicious if: `|residual| > P95` AND enough neighbors
   - Flags anomaly if: `|residual| > P99` AND enough neighbors

2. **Severity Assignment**:
   - Counts P99 anomalies across non-humidity variables
   - Counts P95 suspicious across non-humidity variables
   - Sets severity: `"anomaly"` (1+ P99) → `"suspicious"` (1+ P95) → `"normal"`
   - Humidity anomalies don't trigger alerts independently (they inform but don't drive severity)

### **Key Design Decisions**

- **Humidity special handling**: Higher percentile thresholds (99% vs 95%) because humidity is naturally more variable
- **Non-humidity focus**: Only non-humidity P95/P99 counts drive severity (humidity is informative but not decisive)
- **Minimum neighbors requirement**: Ensures predictions are reliable before flagging
- **Regional vs. local anomalies**: IDW catches isolated spikes (single station deviates) but not coordinated regional events (all stations move together)

### **Test Cases**

`_run_self_test()` validates two scenarios:

- **Case A (Isolated spike)**: Station AWS_A +15°C alone → Flagged as anomaly ✓
- **Case B (Regional event)**: All 4 stations +6°C together → NOT flagged (normal regional variation) ✓

This demonstrates the detector correctly distinguishes sensor malfunctions from legitimate weather patterns.

---

# Anomaly Injector Module

**File**: `anomaly_injector.py`

This file synthetically injects realistic anomalies into clean weather station data for testing detector performance. It creates ground-truth labeled datasets where you know exactly what anomalies were introduced.

## **Core Components**

### **Anomaly Types**

Seven distinct anomaly patterns (as `AnomalyType` enum):

- **SPIKE**: Single sharp deviation from normal
- **OFFSET**: Sustained constant shift in readings
- **DRIFT**: Gradual linear change over time
- **STUCK**: Sensor freezes at one value (zero variance)
- **NOISE**: Increased random fluctuations
- **DROPOUT**: Missing data (NaN values)
- **RATE_CHANGE**: Abrupt change in rate of variation

### **Configuration Classes**

**`AnomalyConfig`**:

- `anomaly_rate`: Target percentage of observations to contaminate (default 1%)
- `min_duration` / `max_duration`: Event length range (1-24 hours, except spikes always = 1)
- `anomaly_type_weights`: Distribution of anomaly types (e.g., spike 20%, offset 20%)
- `severity_weights`: Distribution across low/medium/high (25%/50%/25%)
- `prevent_overlap`: Prevent multiple anomalies on same (station, variable) pair
- `station_local_only`: Constrain anomalies to individual stations

**`AnomalyEvent`**:
Records metadata for each injected anomaly: type, severity, station, variable, time window, magnitude, affected row count.

### **Variable Parameters**

**`VARIABLE_PARAMETERS`** dictionary defines severity-specific magnitudes for each variable:

```
temperature:
  spike: low=(2-4°C), medium=(4-8°C), high=(8-15°C)
  offset: low=(1-2°C), medium=(2-5°C), high=(5-10°C)
  drift: low=(0.05-0.10°C/obs), medium=(0.10-0.30°C/obs), high=(0.30-0.60°C/obs)

humidity: (proportionally larger magnitudes: 8-50% depending on type/severity)
pressure: (smaller magnitudes: 1-10 hPa range)
```

Each variable also has physical bounds (temperature: -10°C to 55°C, humidity: 0-100%, pressure: 950-1060 hPa).

---

## **Injection Functions**

### **Spike** (`inject_spike`)

- Adds random ±magnitude to selected indices
- Duration: 1 observation only
- Clipped to physical range

### **Offset** (`inject_offset`)

- Adds consistent ±magnitude across duration
- Simulates sensor calibration drift
- Example: thermometer reading 2°C high for 6 hours

### **Drift** (`inject_drift`)

- Linear ramp from 0 to ±magnitude over duration
- Simulates gradual sensor degradation
- Uses `np.linspace` to create gradient

### **Stuck** (`inject_stuck`)

- Freezes value at first observation in range
- No magnitude needed (frozen value is constant)
- Detectable by zero variance

### **Noise** (`inject_noise`)

- Scales baseline standard deviation by magnitude factor
- Adds Gaussian noise: `scale = baseline_std × magnitude`
- Simulates sensor calibration loss

### **Dropout** (`inject_dropout`)

- Sets values to NaN
- Simulates sensor failure/missing transmission
- Duration-based (e.g., 4-hour data gap)

### **Rate Change** (`inject_rate_change`)

- Linear ramp added to actual values (not from zero)
- Creates abrupt change in how fast values increase/decrease
- Different from drift (which starts from current baseline)

---

## **Core Injection Logic**

### **`inject_anomalies` (Main Function)**

1. **Validation**: Checks required columns and supported variables

2. **Initialization**:
   - Seeds RNG for reproducibility
   - Creates tracking columns for original values and anomaly metadata
   - Tracks occupied indices to prevent overlap (if enabled)

3. **Anomaly Generation Loop** (up to max_attempts):
   - Randomly select station, variable, anomaly type, severity
   - Determine duration (1 for spikes, 1-24 for others)
   - Find available start position (avoiding occupied regions)
   - Calculate magnitude from variable parameters
   - Inject anomaly using appropriate function
   - Generate unique event_id and record AnomalyEvent

4. **Overlap Prevention**:
   - `_find_available_start`: Searches for duration-length window without conflicts
   - `_mark_occupied`: Tracks which indices are used
   - Prevents same (station, variable) pair from having simultaneous anomalies

5. **Output Columns Added**:
   - `{variable}_original`: Backup of clean values
   - `synthetic_anomaly`: Boolean flag (any anomaly on row)
   - `synthetic_anomaly_type`: Type string (handles multiple overlaps with `|` separator)
   - `synthetic_anomaly_variable`: Variable name
   - `synthetic_anomaly_severity`: Severity level
   - `synthetic_anomaly_event_id`: UUID for event tracking
   - `{variable}_synthetic_anomaly`: Per-variable boolean
   - `{variable}_synthetic_anomaly_type`: Per-variable type

6. **Return**: Contaminated DataFrame + events metadata DataFrame

### **Helper Functions**

- **`_validate_input`**: Checks columns and supported variables
- **`_sample_weighted_choice`**: Probabilistic selection from weighted dict
- **`_sample_magnitude`**: Draws magnitude from range for (variable, type, severity)
- **`_random_sign`**: Returns ±1 randomly
- **`_get_station_indices`**: Builds dict of row indices per station for efficient sampling
- **`_clip_to_physical_range`**: Prevents unrealistic values (e.g., -50°C temperature)

---

## **Statistics & Reporting**

After injection, prints:

- Total/anomalous observation counts
- Actual anomaly rate vs. target
- Number of events created
- Distribution by anomaly type and severity
- Affected observations per variable

Example output:

```
Total observations: 2,304
Anomalous observations: 72
Actual anomaly rate: 3.125%
Anomaly events created: 18
```

---

## **Test Coverage**

`_run_self_test()` validates:

1. ✓ Actual rate matches target (within 1%)
2. ✓ STUCK events have zero variance
3. ✓ DROPOUT events are all NaN
4. ✓ SPIKE events have duration=1
5. ✓ No overlapping events per (station, variable) pair
6. ✓ All values respect physical range bounds

---

## **Key Design Decisions**

- **Severity-based magnitudes**: High severity spikes are more extreme than low severity
- **Variable-specific parameters**: Humidity changes are larger (0-100% scale) than pressure (950-1060 range)
- **Prevent overlap option**: Ensures controlled contamination for evaluation
- **Event tracking**: UUID + metadata enables precise ground truth for testing detectors
- **Original value backup**: Allows contamination rollback if needed
- **Multiple anomalies per observation**: Handles edge cases by pipe-separating values

---

# Statistical Detector Evaluation Module

**File**: `evaluate_statistical.py`

This file comprehensively evaluates the statistical anomaly detector's performance using synthetic ground-truth anomalies. It follows a complete pipeline: calibrate → inject anomalies → detect → analyze → report.

## **High-Level Workflow**

1. **Load datasets**: Historical (calibration) + Present (evaluation)
2. **Calibrate detector**: Learn baselines/thresholds from clean historical data
3. **Inject anomalies**: Add synthetic ground-truth anomalies to evaluation data
4. **Run detector**: Apply statistical detector to contaminated data
5. **Evaluate**: Calculate metrics at observation/event levels
6. **Analyze**: Breakdown by anomaly type, variable, station, severity
7. **Report**: Save CSVs + generate plots

---

## **Core Functions**

### **Data Loading**

**`load_datasets()`**:

- Loads `HISTORICAL_PARQUET` (calibration data, e.g., 30 days clean)
- Loads `PRESENT_PARQUET` (evaluation data, e.g., 7 days for testing)
- Converts timestamps to UTC and sorts by station + time
- Prints summaries: row counts, station counts, date ranges

### **Calibration**

**`calibrate_detector()`**:

- Calls `calibrate_statistical_detector()` on historical data
- Creates `StatisticalConfig` with default parameters
- Returns calibration object + config
- Prints: stations calibrated, z-score windows, percentiles, thresholds

### **Anomaly Injection**

**`inject_evaluation_anomalies()`**:

1. Calls `inject_anomalies()` with 2% target anomaly rate
2. Renames synthetic columns to standard names:
   - `synthetic_anomaly` → `is_anomaly` (ground truth flag)
   - `synthetic_anomaly_event_id` → `anomaly_id` (event tracking)
   - `synthetic_anomaly_type` → `anomaly_type` (spike/drift/etc.)
   - `synthetic_anomaly_variable` → `anomaly_variable` (temp/humidity/pressure)
   - `synthetic_anomaly_severity` → `anomaly_severity` (low/medium/high)
3. Validates all required ground-truth columns exist
4. Saves contaminated data to parquet
5. Reports: total rows, anomalous count, actual rate, event count

### **Detection**

**`run_detector()`**:

- Calls `run_statistical_detector()` on contaminated data
- Returns results with columns: `statistical_alert`, `statistical_severity_label`, etc.
- Prints: total observations, alert count, alert rate percentage

### **Observation-Level Evaluation**

**`calculate_binary_metrics()`**:

- Compares ground truth (`is_anomaly`) vs. predictions (`statistical_alert`)
- Calculates confusion matrix: TP, FP, FN, TN
- Returns:
  - **Precision**: TP / (TP + FP) — of flagged observations, how many are truly anomalous?
  - **Recall**: TP / (TP + FN) — of actual anomalies, how many did we detect?
  - **F1 Score**: Harmonic mean of precision/recall
  - **False Positive Rate**: FP / (FP + TN) — of clean observations, what % did we incorrectly flag?

**`evaluate_observation_level()`**:

- Calls `calculate_binary_metrics()`
- Prints confusion matrix values and all classification metrics
- Returns metrics as DataFrame

### **Event-Level Evaluation**

**`evaluate_event_level()`**:

1. Filters to rows with `is_anomaly=True` (injected anomalies only)
2. Groups by `anomaly_id` (unique event identifier)
3. For each event, calculates:
   - Anomaly type, variable, severity, duration (number of observations)
   - Was event detected? (any row in event has `statistical_alert=True`)
   - How many observations in event were detected?
4. Returns: total events, detected events, missed events, event-level recall
5. **Event-level recall**: What fraction of anomaly _events_ had ≥1 detection?
   - Different from observation-level recall (% of observations detected)

### **Analysis by Anomaly Type**

**`evaluate_by_anomaly_type()`**:

- Filters anomalies only
- Groups by `anomaly_type` (spike, offset, drift, stuck, etc.)
- Calculates recall per type: `detected / total`
- Returns sorted DataFrame showing which anomaly types are easier/harder to detect
- Example: Spikes might have 95% recall, but noise might have 30%

### **Analysis by Variable**

**`evaluate_by_variable()`**:

- Groups injected anomalies by variable (temperature, humidity, pressure)
- Calculates recall per variable
- Shows which variables are detected better
- Example: Temperature anomalies might have higher recall than humidity

### **False Positive Analysis**

**`analyze_false_positives()`**:

1. Filters to clean observations only (`is_anomaly=False`)
2. Counts how many were flagged as alerts (`statistical_alert=True`)
3. Calculates FP rate: `false_positives / total_clean_observations`
4. Breaks down FP counts by station
5. Identifies stations with high false positive rates (over-sensitive)

### **Detector Firing Rates**

**`analyze_detector_firing_rates()`**:

- Lists individual detector columns: `temperature_range_anomaly`, `humidity_zscore_6_anomaly`, etc.
- Counts how often each detector fires
- Calculates firing rate percentage
- Sorted by frequency (most-firing first)
- Example:
  ```
  detector                        alert_count  alert_rate_percent
  temperature_range_anomaly       145          6.29%
  humidity_zscore_24_anomaly      89           3.86%
  pressure_persistence_anomaly    12           0.52%
  ```

### **Evidence Family Analysis**

**`analyze_evidence_families()`**:

- Groups detector outputs into four families: range, ROC, level (z-score), persistence
- Shows how many observations triggered each family
- Identifies which evidence type is most common
- Example: "range" family might fire on 8.2% of observations, "level" on 2.1%

### **Station Analysis**

**`analyze_station_alerts()`**:

- Groups by `station_id`
- Calculates: total observations, alert count, alert rate
- Shows which stations have highest alert rates (might indicate sensor issues or installation location)

### **Severity Distribution**

**`analyze_severity_distribution()`**:

- Groups by `statistical_severity_label` (normal, suspicious, anomaly, critical)
- Counts observations and injected anomalies per severity level
- Shows: how many anomalies fell into each bucket?
- Example: 45 critical (all injected), 89 anomaly (78 injected), 1230 suspicious (2 injected)

### **Alert Summary**

**`generate_alert_summary()`**:

- Calls `summarize_statistical_alerts()` to aggregate detector signals
- Removes synthetic columns, keeps real detector signals
- Sorts by alert count
- Returns DataFrame with variable, detector, count, rate

---

## **Plotting**

**`generate_plots()`**:
Generates 4-panel figure saved to `statistical_detector_evaluation.png`:

1. **Confusion Matrix** (top-left):
   - Heatmap showing TP, FP, FN, TN counts
   - Visual summary of classification performance

2. **Recall by Anomaly Type** (top-right):
   - Bar chart: anomaly type vs. recall %
   - Shows which synthetic anomalies are detected well

3. **Alert Rate by Station** (bottom-left):
   - Horizontal bar chart: station ID vs. alert rate %
   - Identifies stations with high/low alert rates

4. **Severity Distribution** (bottom-right):
   - Bar chart: severity label vs. observation count
   - Shows distribution of normal/suspicious/anomaly/critical

---

## **Results Saving**

**`save_results()`** saves 10 output files to `RESULTS_STATISTICAL_DIR`:

| File                                  | Content                                                            |
| ------------------------------------- | ------------------------------------------------------------------ |
| `observation_metrics.csv`             | TP, FP, FN, TN, precision, recall, F1, FPR                         |
| `event_metrics.csv`                   | Per-event detection: type, variable, severity, duration, detected? |
| `performance_by_anomaly_type.csv`     | Recall for spike/offset/drift/stuck/noise/dropout/rate_change      |
| `performance_by_variable.csv`         | Recall for temp/humidity/pressure                                  |
| `false_positive_analysis.csv`         | FP counts per station                                              |
| `detector_firing_rates.csv`           | Individual detector signal rates                                   |
| `evidence_family_summary.csv`         | Range/ROC/level/persistence firing rates                           |
| `station_statistics.csv`              | Alert rates per station                                            |
| `severity_distribution.csv`           | Distribution across normal/suspicious/anomaly/critical             |
| `injection_log.csv`                   | Details of all injected anomaly events                             |
| `anomaly_predictions.csv`             | Rows with `is_anomaly=True` OR `statistical_alert=True`            |
| `full_results.parquet`                | Complete result DataFrame (all columns)                            |
| `statistical_detector_evaluation.png` | 4-panel evaluation plot                                            |

---

## **Key Metrics Explained**

**Precision vs. Recall Trade-off**:

- **High Precision, Low Recall**: Detector is conservative, rarely flags things. When it does, it's usually right. But misses many real anomalies.
- **Low Precision, High Recall**: Detector is aggressive, catches most real anomalies but produces many false alarms.
- **Goal**: Balance both (high F1 score)

**Event-Level vs. Observation-Level**:

- **Event-level recall**: Did we detect the anomaly _somewhere_ in its duration?
- **Observation-level recall**: What percentage of individual anomalous readings did we catch?
- Event-level is more lenient (1 detection = success for whole event)

**False Positive Rate**:

- Ratio of incorrectly flagged clean observations
- Example: 0.01% FPR = 1 in 10,000 clean observations falsely flagged
- Critical for operational systems (high FP → alert fatigue)

---

## **Example Output Flow**

```
Load datasets:
  Historical: 50,000 rows, 10 stations, 2024-11-01 to 2024-11-30
  Present: 7,000 rows, 10 stations, 2024-12-01 to 2024-12-07

Calibrate detector:
  ✓ 10 stations calibrated
  Z-score windows: [24, 168] hours
  ROC percentile: 99.0%

Inject anomalies:
  Total observations: 7,000
  Anomalous: 140 (2.0%)
  Events created: 35

Run detector:
  Alerts: 125 / 7,000 (1.79%)

Observation-level:
  Precision: 0.92 (125 flagged, 115 truly anomalous)
  Recall: 0.82 (140 real anomalies, 115 detected)
  F1: 0.87

Event-level:
  Detected: 32 / 35 events (91.4% recall)

By anomaly type:
  Spike: 95% recall
  Offset: 87%
  Drift: 78%
  Stuck: 100%
  Noise: 25%
```

This modular evaluation framework enables comprehensive assessment of detector strengths and weaknesses.

---

# Spatial Detector Evaluation Module

**File**: `evaluate_spatial.py`

This file comprehensively evaluates the spatial anomaly detector's performance using the same methodology as the statistical evaluator, but tailored for spatial consistency checks (comparing stations to neighbors).

---

## **High-Level Workflow**

1. **Load datasets**: Historical (calibration) + Present (evaluation)
2. **Calibrate detector**: Build neighbor network and residual thresholds from clean data
3. **Inject anomalies**: Add synthetic ground-truth anomalies to evaluation data
4. **Run detector**: Apply spatial detector to contaminated data
5. **Evaluate**: Calculate metrics at observation/event/anomaly-type levels
6. **Test regional events**: Sanity check that shared events aren't misclassified
7. **Report**: Save CSVs + generate plots

---

## **Core Functions**

### **Data Loading**

**`load_datasets()`**:

- Loads `HISTORICAL_PARQUET` and `PRESENT_PARQUET` (same as statistical evaluator)
- Ensures timestamps are UTC
- Prints summaries: row counts, station counts, date ranges
- Returns both DataFrames

### **Calibration**

**`calibrate_detector()`**:

1. Extracts station coordinates (latitude, longitude) from data
2. Calls `calibrate_spatial_detector()` with:
   - Historical data
   - Variables: temperature, humidity, pressure
   - Station coordinates dict
   - Default `SpatialConfig` (k_neighbors=4, idw_power=2.0)
3. Returns calibration object + config
4. Prints: "Spatial calibration complete"

Key difference from statistical: **requires geographic coordinates** to build neighbor network.

### **Anomaly Injection**

**`inject_evaluation_anomalies()`**:

- Same as statistical evaluator
- 2% target anomaly rate
- Renames synthetic columns:
  - `synthetic_anomaly` → `is_anomaly`
  - `synthetic_anomaly_event_id` → `anomaly_id`
  - `synthetic_anomaly_type` → `anomaly_type`
  - `synthetic_anomaly_variable` → `anomaly_variable`
  - `synthetic_anomaly_severity` → `anomaly_severity`
- Saves to `SPATIAL_EVAL_INJECTED_PARQUET`
- Prints: total observations, anomalous count, actual rate

### **Detection**

**`run_detector()`**:

- Calls `run_spatial_detector()` on contaminated data
- Returns results with `spatial_alert` column
- Prints: total observations, alert count, alert rate

---

## **Performance Evaluation Functions**

### **Binary Classification Metrics**

**`calculate_binary_metrics()`**:

- Same as statistical evaluator
- Compares ground truth (`is_anomaly`) vs. predictions (`spatial_alert`)
- Calculates: TP, FP, FN, TN, precision, recall, F1, false positive rate

### **Observation-Level Evaluation**

**`evaluate_observation_level()`**:

- Calls `calculate_binary_metrics()`
- Prints confusion matrix: TP, FP, FN, TN counts
- Prints classification metrics: precision, recall, F1, FPR
- Returns metrics as DataFrame

### **Event-Level Evaluation**

**`evaluate_event_level()`**:

1. Filters to injected anomalies only (`is_anomaly=True`)
2. Groups by `anomaly_id` (unique event)
3. Aggregates per event:
   - `anomaly_type`, `variable`, `duration` (# observations in event)
   - `detected`: Was ≥1 observation in event flagged?
   - `observations_detected`: Count of detected observations in event
4. Calculates **event-level recall**: (detected events) / (total events)
5. Returns DataFrame with one row per anomaly event

Example:

```
anomaly_id  anomaly_type  variable     duration  detected  observations_detected
evt_001     spike         temperature  1         True      1
evt_002     drift         humidity     12        True      8
evt_003     offset        pressure     6         False     0
```

### **Performance by Anomaly Type**

**`evaluate_by_anomaly_type()`**:

1. Filters to injected anomalies only
2. Groups by `anomaly_type` (spike, offset, drift, stuck, noise, dropout, rate_change)
3. Per type, calculates:
   - Total observations
   - Detected observations
   - Missed observations
   - **Recall**: detected / total
4. Sorted by recall (descending)

Shows which anomaly types spatial detector is good/bad at detecting.

Example output:

```
anomaly_type  total_observations  detected  missed  recall  recall_percent
stuck         45                  44        1       0.9778  97.78
spike         52                  48        4       0.9231  92.31
offset        38                  30        8       0.7895  78.95
drift         41                  28        13      0.6829  68.29
rate_change   35                  18        17      0.5143  51.43
noise         33                  10        23      0.3030  30.30
```

### **Performance by Variable**

**`evaluate_by_variable()`**:

1. For each variable (temperature, humidity, pressure)
2. Filters to rows where:
   - Ground truth: `is_anomaly=True` AND `anomaly_variable==variable`
   - Prediction: `{variable}_spatial_anomaly=True`
3. Calculates binary metrics per variable
4. Returns DataFrame with one row per variable

Shows which variables (temp/humidity/pressure) spatial detector handles best.

### **False Positive Analysis**

**`analyze_false_positives()`**:

1. Filters to clean observations (`is_anomaly=False`)
2. Counts how many were incorrectly flagged as alerts (`spatial_alert=True`)
3. Calculates false positive rate: FP / (FP + TN)
4. Breaks down FP counts by station
5. Prints: total clean, FP count, FP rate, FP by station

Identifies stations with high false positive rates (over-sensitive or noisy).

### **Severity Distribution**

**`analyze_severity_distribution()`**:

1. Groups by `spatial_severity_level` (normal, suspicious, anomaly, critical)
2. For each severity level, calculates:
   - Total observations
   - Injected anomalies in that level
   - Alerts in that level
   - Anomaly rate %
3. Returns DataFrame

Shows distribution of observations across severity buckets.

Example:

```
spatial_severity_level  observations  anomalies  alerts  anomaly_rate_percent
normal                  4500          5          0       0.11
suspicious              800           20         5       2.50
anomaly                 150           110        108     73.33
critical                0             0          0       —
```

---

## **Sanity Test: Regional Event**

**`run_regional_event_test()`**:

This is a critical **sanity check** that spatial detector correctly distinguishes:

- **Isolated anomalies** (one station deviates) → Should be flagged ✓
- **Regional events** (all stations move together) → Should NOT be flagged ✓

Process:

1. Randomly selects a timestamp in middle of dataset
2. Adds shared +6°C to all stations at that timestamp (simulating regional weather event)
3. Runs spatial detector on modified data
4. Counts alerts during regional event
5. **Expected**: 0 alerts (all stations agree, so IDW interpolation matches)
6. **Result**: Prints PASS/WARNING

This validates that detector catches sensor malfunctions but ignores real weather patterns.

Example output:

```
Injected regional event:
  Timestamp: 2026-12-15 14:00:00
  Variable: temperature
  Shared offset: +6.0
  Stations affected: 10

Spatial alerts during regional event: 0/10
✓ PASS — regional event was not misclassified.
```

---

## **Plotting**

**`generate_plots()`** creates 4-panel figure (`spatial_detector_evaluation.png`):

1. **Confusion Matrix** (top-left):
   - 2×2 heatmap: TN, FP, FN, TP counts
   - Shows classification performance at a glance

2. **Recall by Anomaly Type** (top-right):
   - Bar chart: anomaly_type vs. recall %
   - Example: spike=92%, drift=68%, noise=30%
   - Shows which anomaly types are easier/harder

3. **Alert Rate by Station** (bottom-left):
   - Horizontal bar chart: station_id vs. alert rate %
   - Identifies stations with high/low alert rates
   - May indicate sensor issues or geographic patterns

4. **Severity Distribution** (bottom-right):
   - Bar chart: severity_level (normal/suspicious/anomaly/critical) vs. observation count
   - Shows how observations are distributed across severity buckets

---

## **Results Saving**

**`save_results()`** saves 8 output files to `OUTPUT_DIR` (results/spatial/):

| File                              | Content                                                       |
| --------------------------------- | ------------------------------------------------------------- |
| `injection_log.csv`               | Details of all injected anomaly events                        |
| `observation_metrics.csv`         | TP, FP, FN, TN, precision, recall, F1, FPR                    |
| `event_metrics.csv`               | Per-event detection: type, variable, duration, detected?      |
| `performance_by_anomaly_type.csv` | Recall for spike/offset/drift/stuck/noise/dropout/rate_change |
| `performance_by_variable.csv`     | Per-variable metrics (temp/humidity/pressure)                 |
| `false_positive_analysis.csv`     | FP counts per station                                         |
| `severity_distribution.csv`       | Distribution across normal/suspicious/anomaly/critical        |
| `regional_event_test.csv`         | Results from sanity test (stations during regional event)     |
| `anomaly_predictions.csv`         | Rows with `is_anomaly=True` OR `spatial_alert=True`           |
| `full_results.parquet`            | Complete result DataFrame (all columns)                       |
| `spatial_detector_evaluation.png` | 4-panel evaluation plot                                       |

---

## **Main Execution**

**`main()`** orchestrates the complete pipeline:

```
1. Load datasets
2. Calibrate spatial detector
3. Inject synthetic anomalies
4. Run spatial detector on corrupted data
5. Evaluate at observation level
6. Evaluate at event level
7. Evaluate by anomaly type
8. Evaluate by variable
9. Analyze false positives
10. Analyze severity distribution
11. Run regional event sanity test
12. Generate plots
13. Save all results
14. Print summary
```

Prints timestamps and progress messages throughout.

---

## **Key Differences from Statistical Evaluator**

| Aspect                | Statistical                          | Spatial                               |
| --------------------- | ------------------------------------ | ------------------------------------- |
| **Calibration data**  | Time series patterns                 | Geographic coordinates                |
| **Detection logic**   | Temporal baselines (hourly)          | Spatial interpolation (IDW)           |
| **Anomaly detection** | Per-variable deviation from baseline | Per-station deviation from neighbors  |
| **Regional events**   | May flag all stations                | Should NOT flag (all neighbors agree) |
| **Sanity test**       | Isolated spike vs. noise             | Regional event vs. local spike        |
| **Strength**          | Detects temporal patterns            | Detects spatial inconsistencies       |
| **Weakness**          | Misses coordinated regional events   | Misses coordinated regional events    |

---

## **Example Evaluation Output**

```
SPATIAL DETECTOR EVALUATION
Started: 2026-09-01 14:30:00

LOADING DATASETS
Historical calibration data:
  Rows:     50,000
  Stations: 10
  Period:   2026-08-01 to 2026-08-31

Unseen evaluation data:
  Rows:     7,000
  Stations: 10
  Period:   2026-09-01 to 2026-09-07

CALIBRATING SPATIAL DETECTOR
✓ Spatial calibration complete

INJECTING SYNTHETIC ANOMALIES
Total observations:      7,000
Anomalous observations:  140
Actual anomaly rate:     2.00%

RUNNING SPATIAL DETECTOR
Total observations: 7,000
Spatial alerts:     95
Alert rate:         1.357%

OBSERVATION-LEVEL PERFORMANCE
True Positives:  87
False Positives: 8
False Negatives: 53
True Negatives:  6,852

Precision:           0.9157
Recall:              0.6214
F1 Score:            0.7407
False Positive Rate: 0.0012

EVENT-LEVEL PERFORMANCE
Total anomaly events:    35
Detected anomaly events: 30
Missed anomaly events:   5
Event-level recall:      0.8571

PERFORMANCE BY ANOMALY TYPE
anomaly_type  total_observations  detected  missed  recall  recall_percent
stuck         44                  44        0       1.0000  100.00
spike         52                  50        2       0.9615  96.15
offset        38                  28        10      0.7368  73.68
drift         41                  22        19      0.5366  53.66
noise         33                  8         25      0.2424  24.24

FALSE POSITIVE ANALYSIS
Clean observations: 6,860
False positives:    8
False positive rate: 0.1166%

REGIONAL EVENT SANITY TEST
Injected regional event:
  Timestamp: 2026-09-04 12:00:00
  Variable: temperature
  Shared offset: +6.0
  Stations affected: 10

Spatial alerts during regional event: 0/10
✓ PASS — regional event was not misclassified.

Spatial evaluation complete: 2026-09-01 14:35:30
Results saved to: /home/aaryan/sih/skyguard/results/spatial
```

---

## **Interpretation Guide**

- **High Precision, Low Recall**: Detector is conservative; few false alarms but misses anomalies
- **Low Precision, High Recall**: Detector is aggressive; catches most anomalies but many false alarms
- **F1 Score**: Balanced metric between precision and recall (target: >0.8)
- **Event-level recall > Observation-level recall**: Normal (catching ≥1 observation per event is easier than catching all)
- **Spike/Stuck detection >95%**: Spatial detector excellent for obvious local anomalies
- **Drift detection <70%**: Harder to detect gradual changes (may blend with neighbors)
- **Regional event test PASS**: Detector correctly ignores shared weather patterns

---

## End of Documentation

This comprehensive guide covers all major components of the SkyGuard anomaly detection system for weather station observations. Each module is designed to work together: statistical detection identifies temporal anomalies, spatial detection identifies geographic inconsistencies, anomaly injection creates ground-truth datasets, and evaluation frameworks assess detector performance comprehensively.

---

# Prototype Integration Reference

The prototype adds an offline preparation and runtime boundary around the
existing detectors. The dashboard consumes precomputed results and does not
run calibration or detector inference on startup.

## Final Demo Dataset

`src/skyguard/simulation/create_final_dataset.py` creates the reproducible
2026 benchmark from the clean present dataset. It uses:

```text
anomaly rate: 2%
random seed: 42
variables: temperature, humidity, pressure
```

Outputs:

```text
data/synthetic/skyguard_demo_2026.parquet
data/synthetic/skyguard_demo_2026_log.parquet
```

The dataset retains `synthetic_*` metadata and canonical evaluation aliases:
`is_anomaly`, `anomaly_id`, `anomaly_type`, `anomaly_variable`, and
`anomaly_severity`. These fields are for evaluation and explanation only.

Generate it with:

```bash
python -m skyguard.simulation.create_final_dataset
```

## Calibration Artifacts

`src/skyguard/fusion/calibrate.py` calibrates statistical and spatial
detectors from clean historical data only, covering 2023-01-01 through
2025-12-31. It copies the existing frozen LSTM calibration state without
retraining it.

Outputs:

```text
artifacts/skyguard_v1/statistical.pkl
artifacts/skyguard_v1/spatial.pkl
artifacts/skyguard_v1/lstm.pkl
artifacts/skyguard_v1/fusion_config.json
```

Run it with:

```bash
python -m skyguard.fusion.calibrate
```

The injected 2026 dataset and its synthetic labels must not be used by this
calibration step.

## Runtime Fusion Contract

`src/skyguard/fusion/fusion.py` exposes:

```python
results = run_fusion(data)
```

The input requires:

```text
station_id
timestamp
temperature
pressure
humidity
```

The function loads the saved artifacts, runs the three frozen detectors, and
applies the deterministic 2-of-3 policy:

```python
detector_votes = (
   statistical_alert.astype(int)
   + spatial_alert.astype(int)
   + lstm_ae_alert.astype(int)
)
final_alert = detector_votes >= 2
```

The output retains input fields and detector-prefixed fields, then adds:

```text
detector_votes
final_alert
final_severity
final_confidence
```

Final severity is mapped from detector agreement:

```text
0 -> normal
1 -> suspicious
2 -> anomaly
3 -> critical
```

`final_confidence` is `detector_votes / 3`, representing agreement rather
than a calibrated probability. Runtime decisions do not read synthetic
ground-truth columns.

## Precomputed Prototype Results

The one-time inference command is:

```bash
python -m skyguard.fusion.fusion
```

It reads the final injected benchmark and writes:

```text
results/prototype/skyguard_demo_2026_results.parquet
```

Streamlit reads this result file for replay, station status, detector
agreement, severity, and alert history. The dashboard must not retrain,
recalibrate, inject anomalies, or run detector inference during startup.
