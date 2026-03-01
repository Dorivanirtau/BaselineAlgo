# Suicide Risk Prediction Pipeline - Documentation

## Overview

This project implements a passive suicide risk prediction pipeline that uses smartphone sensor data to predict psychological constructs (restlessness, impulsivity, irritability, insomnia) and detect deviations from a student's personal baseline. When a student's daily risk score deviates significantly from their norm, the system flags an elevated risk alert — which in a real deployment would trigger a Voice Agent check-in call.

The system operates in **two distinct phases:**

1. **Training Phase (one-time):** Uses a historical dataset of sensor readings + EMA survey responses to train 4 Random Forest models. This only needs to happen once.
2. **Baseline/Deployment Phase (ongoing, daily):** Each day, **all raw sensor data** is collected from the student's smartphone, processed into the 20 engineered features, and fed into the trained models. The system then computes a composite suicide risk score, compares it against the student's personal 14-day rolling baseline, and triggers an alert if the score deviates significantly.

---

## The 3 Clinical Stages

The pipeline mirrors 3 real-world clinical stages of SSRI treatment monitoring:

```
Stage 1 (Training)          Stage 2 (Baseline)          Stage 3 (Monitoring)
   Historical data    -->    Patient's first 14 days  -->  Ongoing daily data
   Learns GENERAL            Learns THIS PATIENT's        Detects DEVIATIONS
   sensor-construct          personal normal              from personal normal
   relationships             behavior                     after SSRI starts
        |                         |                            |
        v                         v                            v
   Trained models            Personal baseline            ALERT if z > 1.5
   (reusable for             (unique per patient)         --> Voice Agent call
    all patients)
```

### Stage 1: Training (Historical Dataset)
**Purpose:** Learn *general patterns* between sensor behavior and psychological constructs.
- **Input:** The StudentLife dataset (historical sensor data + EMA survey responses from 49 students)
- **Output:** 4 trained Random Forest models (`models/*.pkl`)
- **When:** One-time, done offline before deploying to any patient
- **Scripts:** `sensor_features.py` (Step 1) + `train_models.py` (Step 2)

This stage teaches the models: *"what does restlessness look like in sensor data?"*, *"what does insomnia look like?"*, etc. The models learn these relationships from the historical population and are then reusable for all future patients.

### Stage 2: Baseline Period (Before/At Start of SSRI)
**Purpose:** Establish the patient's **personal norm** — what their "normal" behavior looks like.
- **Input:** ~14 days of raw sensor data from the patient's phone (all 9 sensor sources collected daily)
- **Output:** A personal rolling baseline of their suicide risk score
- **When:** The first 14 days of monitoring (before or just as the patient starts SSRI treatment)

During this period, all raw sensors (phonelock, dark, GPS, conversation, call log, SMS, phonecharge, WiFi location, Bluetooth) are collected every day and scored by the trained models, but **no alerts are triggered yet**. The system is simply learning: *"this is what THIS specific patient's normal looks like."*

### Stage 3: Post-SSRI Monitoring (Active Detection)
**Purpose:** Detect behavioral *deviations* from the personal baseline that may indicate SSRI-induced suicidality.
- **Input:** Continued daily raw sensor data from the patient's phone (same 9 sensor sources)
- **Output:** Elevated risk alerts when z-score > 1.5
- **When:** Ongoing, every day after the baseline period
- **Script:** `alert_system.py` (Step 4)

This is where alerts fire. The system compares each new day against the baseline established in Stage 2 and asks: *"is this patient's behavior significantly different from their own norm?"* If yes — for example, sudden insomnia, social isolation, or restless phone usage — the system triggers an elevated risk alert, which would initiate a Voice Agent check-in call.

### Why Personal Baselines Matter
Stage 1 models are **general** (trained once, used for all patients), but Stage 2 baselines are **personal** (unique to each patient). Stage 3 detects changes by comparing against each patient's *own* norm — not a population average. This is critical because what is "normal" for one person may be abnormal for another (e.g., one patient may naturally have high night phone usage while another does not).

---

## Pipeline Architecture

### Training Phase (One-Time Setup)

```
Historical Sensor CSVs       PCA Construct Scores (from EMA surveys)
    (dataset/sensing/)           (weight_features/*_scores.csv)
           |                                |
           v                                v
     [Step 1]                         (Pre-computed)
     sensor_features.py               Ground truth Y labels
           |                                |
           v                                v
     sensor_daily_features.csv              |
           |________________________________|
                        |
                        v
                  [Step 2]
                  train_models.py
                        |
                        v
                  models/*.pkl (4 trained Random Forest models)
```

### Baseline/Deployment Phase (Daily, Per Student)

```
  Student's smartphone (daily)
           |
           | Collects ALL raw sensors:
           | phonelock, dark, GPS, conversation,
           | call_log, SMS, phonecharge,
           | wifi_location, bluetooth
           |
           v
     [Feature Engineering]
     Same logic as Step 1: aggregate raw
     sensor data into 20 daily features
           |
           v
     20 engineered features for today
           |
           v
     [Step 3/4: Scoring & Alert]
     Feed into 4 trained Random Forest models
           |
           v
     4 predicted construct scores
     (restlessness, impulsivity, irritability, insomnia)
           |
           v
     Composite suicide_risk_score
     + sensor_anomaly_score
           |
           v
     Compare to personal 14-day rolling baseline
           |
           v
     z_score > 1.5? --> ELEVATED RISK ALERT
                        (trigger Voice Agent check-in)
```

### Whole-System Input/Output

| | Training Phase | Baseline/Deployment Phase |
|---|---|---|
| **Input** | Historical sensor CSVs (`dataset/sensing/`, `dataset/call_log/`, `dataset/sms/`) + PCA construct score CSVs (`weight_features/*_scores.csv`) from EMA surveys | **Daily raw sensor data** from the student's smartphone: phonelock, dark, GPS, conversation, call log, SMS, phone charge, WiFi location, Bluetooth — all 9 sensor sources are ingested each day |
| **Output** | 4 trained model files (`models/*.pkl`) + `sensor_daily_features.csv` | Per-day risk assessment: 4 construct predictions, composite `suicide_risk_score`, personal `baseline`, `z_score`, and `elevated_risk` flag (0/1). If elevated, triggers Voice Agent intervention |
| **Frequency** | One-time | Every day, per student |

---

## Step 1: Sensor Feature Engineering

**Script:** `weight_features/sensor_features.py`  
**Run:** `python weight_features/sensor_features.py`

### Purpose
Converts raw sensor CSV files into a single structured table of daily aggregated features per student.

### Input

| Source | Path Pattern | Key Columns |
|---|---|---|
| Phone Lock | `dataset/sensing/phonelock/phonelock_u{XX}.csv` | `start`, `end` (unix timestamps) |
| Dark/Screen Off | `dataset/sensing/dark/dark_u{XX}.csv` | `start`, `end` (unix timestamps) |
| GPS | `dataset/sensing/gps/gps_u{XX}.csv` | `time`, `latitude`, `longitude` |
| Conversation | `dataset/sensing/conversation/conversation_u{XX}.csv` | `start_timestamp`, `end_timestamp` |
| Call Log | `dataset/call_log/call_log_u{XX}.csv` | `timestamp`, `CALLS_type`, `CALLS_duration` |
| SMS | `dataset/sms/sms_u{XX}.csv` | `timestamp`, `MESSAGE_type` |
| Phone Charge | `dataset/sensing/phonecharge/phonecharge_u{XX}.csv` | `start`, `end` |
| WiFi Location | `dataset/sensing/wifi_location/wifi_location_u{XX}.csv` | `time`, `location` |
| Bluetooth | `dataset/sensing/bluetooth/bt_u{XX}.csv` | `time`, `MAC` |

Users processed: `u00` through `u59` (60 possible, ~49 have data).

### Processing Logic

For **each student**, for **each day**, the script aggregates:

| Sensor | Features Extracted | Description |
|---|---|---|
| phonelock | `unlock_count` | Number of phone unlock events |
| | `avg_session_sec` | Average duration of each unlocked session (seconds) |
| | `total_unlocked_sec` | Total time phone was unlocked (seconds) |
| | `night_unlocks` | Number of unlocks between 00:00-06:00 |
| dark | `total_dark_hrs` | Total hours screen was off |
| | `dark_fragments` | Number of separate dark periods |
| | `longest_dark_streak_hrs` | Longest continuous dark period (hours) |
| | `night_dark_hrs` | Dark hours during 00:00-06:00 (proxy for sleep) |
| gps | `total_distance_km` | Total distance travelled (haversine formula) |
| | `num_clusters` | Number of unique locations visited (rounded to 3 decimals) |
| | `location_entropy` | Shannon entropy of location distribution (higher = more varied) |
| | `time_at_top_location_pct` | Percentage of time at most visited location |
| conversation | `convo_count` | Number of face-to-face conversations detected |
| | `total_convo_min` | Total conversation duration (minutes) |
| | `avg_convo_length_min` | Average conversation length (minutes) |
| call_log | `incoming_calls` | Count of incoming calls |
| | `outgoing_calls` | Count of outgoing calls |
| | `missed_calls` | Count of missed calls |
| | `total_call_min` | Total call duration (minutes) |
| sms | `sms_sent` | Count of SMS messages sent |
| | `sms_received` | Count of SMS messages received |
| phonecharge | `charge_sessions` | Number of charging sessions |
| | `night_charge_hrs` | Hours spent charging during 00:00-06:00 |
| wifi_location | `unique_locations` | Number of distinct WiFi location labels |
| | `top_location_pct` | Percentage of time at most common WiFi location |
| bluetooth | `avg_nearby_devices` | Average number of nearby Bluetooth devices per scan |

### Missing Data Handling
1. If a sensor has no data for a day, features are set to `NaN`
2. Days with >50% `NaN` features are dropped entirely
3. Remaining `NaN` values are filled with the student's personal median
4. Any still-remaining `NaN` values are filled with 0

### Output

| File | Shape | Format |
|---|---|---|
| `sensor_daily_features.csv` | 2,683 rows x 22 columns | CSV |

**Columns:** `uid`, `date`, + 20 feature columns listed above

---

## Step 2: Model Training

**Script:** `train_models.py`  
**Run:** `python train_models.py`

### Purpose
Trains 4 separate Random Forest Regressor models, each mapping sensor features (X) to a psychological construct score (Y).

### Input

| Input | Source | Description |
|---|---|---|
| X features | `sensor_daily_features.csv` | 20 sensor features per student per day (from Step 1) |
| Y labels (restlessness) | `weight_features/restlessness_scores.csv` | PCA-derived restlessness score per student per day |
| Y labels (impulsivity) | `weight_features/impulsivity_scores.csv` | PCA-derived impulsivity score per student per day |
| Y labels (irritability) | `weight_features/irritability_scores.csv` | PCA-derived irritability score per student per day |
| Y labels (insomnia) | `weight_features/insomnia_scores.csv` | PCA-derived insomnia score per student per day |

The Y-label CSVs have columns: `date`, component value columns, `student`, `{construct}_score`.  
Merging is performed on `uid = student` and `date = date` (inner join).

### Processing Logic
For each of the 4 constructs:
1. **Merge** X and Y on `uid`/`student` + `date` (inner join — only days with both sensor data and survey data)
2. **Drop** rows where the target score is NaN
3. **Evaluate** using 5-Fold Cross Validation with metrics: MAE, RMSE, R-squared
4. **Train** a `RandomForestRegressor(n_estimators=100)` on the full dataset
5. **Save** the trained model as a `.pkl` file

### Output

| File | Description |
|---|---|
| `models/restlessness_rf.pkl` | Trained RF model: sensors -> restlessness score |
| `models/impulsivity_rf.pkl` | Trained RF model: sensors -> impulsivity score |
| `models/irritability_rf.pkl` | Trained RF model: sensors -> irritability score |
| `models/insomnia_rf.pkl` | Trained RF model: sensors -> insomnia score |

**Console output:** MAE, RMSE, R-squared for each model.

### Model Performance (5-Fold CV)

| Construct | MAE | RMSE | R-squared |
|---|---|---|---|
| Restlessness | 14.64 | 18.25 | -0.09 |
| Impulsivity | 14.27 | 17.94 | -0.09 |
| Irritability | 16.95 | 21.51 | -0.05 |
| Insomnia | 14.76 | 18.58 | -0.02 |

---

## Step 3: Batch Risk Scoring & Deviation Detection

**Script:** `predict_risk.py`  
**Run:** `python predict_risk.py`

### Purpose
Runs all sensor data through the 4 trained models to produce daily risk scores, computes personal baselines, and flags deviations.

### Input

| Input | Source |
|---|---|
| `sensor_daily_features.csv` | 20 daily sensor features per student (from Step 1) |
| `models/*.pkl` | 4 trained RF models (from Step 2) |

### Processing Logic (6 sub-steps)

| Sub-step | Action | Output |
|---|---|---|
| 1. Load Features | Read `sensor_daily_features.csv` | DataFrame with 2,683 rows x 22 cols |
| 2. Predict Constructs | Run each row through 4 RF models | 4 new columns: `pred_restlessness`, `pred_impulsivity`, `pred_irritability`, `pred_insomnia` |
| 3. Composite Score | `suicide_risk_score = mean(4 predictions)` | 1 new column |
| 4. Personal Baseline | Rolling 14-day mean of `suicide_risk_score` per student | `baseline` column |
| 5. Deviation Detection | `z_score = (score - baseline) / rolling_std`; if z > 1.5 then `elevated_risk = 1` | `z_score` and `elevated_risk` columns |
| 6. Save | Write final DataFrame to CSV | `daily_suicide_risk.csv` |

### Output

| File | Shape | Format |
|---|---|---|
| `daily_suicide_risk.csv` | 2,683 rows x 10 columns | CSV |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `uid` | string | Student identifier (e.g. `u00`) |
| `date` | datetime | Calendar date |
| `pred_restlessness` | float | Predicted restlessness score for the day |
| `pred_impulsivity` | float | Predicted impulsivity score for the day |
| `pred_irritability` | float | Predicted irritability score for the day |
| `pred_insomnia` | float | Predicted insomnia score for the day |
| `suicide_risk_score` | float | Composite risk = mean of 4 predictions |
| `baseline` | float | Student's personal rolling 14-day mean |
| `z_score` | float | How many std deviations above/below baseline |
| `elevated_risk` | int (0/1) | 1 if z_score > 1.5, else 0 |

---

## Step 4: Real-Time Alert System

**Script:** `alert_system.py`  
**Run:** `python alert_system.py`

### Purpose
Simulates a real-time monitoring system that processes one day of sensor data at a time and determines whether to trigger a clinical intervention alert.

### Input (per call)

| Input | Description |
|---|---|
| `uid` | Student identifier |
| `date` | The current date |
| `sensor_dict` | Dictionary of 20 sensor feature values for that day |

The `StudentMonitor` class also internally maintains:
- `sensor_history`: list of all past days' raw sensor readings (for anomaly detection)
- `history`: list of all past days' composite risk scores (for baseline/z-score)

### Processing Logic (per day)

```
New sensor data (dict of 20 features)
         |
         v
  [1] Predict 4 construct scores via trained RF models
         |
         v
  [2] Compute model_risk = mean(4 predictions)
         |
         v
  [3] Compute sensor_anomaly = how unusual today's
      raw sensors are vs student's own history
      (average z-score across all 20 features, scaled 0-100)
         |
         v
  [4] Composite: suicide_risk_score = 50% model_risk + 50% sensor_anomaly
         |
         v
  [5] Compare to rolling 14-day personal baseline
      z_score = (score - baseline) / rolling_std
         |
         v
  [6] If z_score > 1.5 --> ELEVATED RISK ALERT
      (trigger Voice Agent check-in)
```

### Output (per call)

| Field | Type | Description |
|---|---|---|
| `uid` | string | Student identifier |
| `date` | string | Date processed |
| `pred_restlessness` | float | Predicted restlessness |
| `pred_impulsivity` | float | Predicted impulsivity |
| `pred_irritability` | float | Predicted irritability |
| `pred_insomnia` | float | Predicted insomnia |
| `model_risk` | float | Mean of 4 model predictions |
| `sensor_anomaly` | float | Sensor anomaly score (0-100) |
| `suicide_risk_score` | float | Blended composite risk |
| `baseline` | float | Rolling 14-day personal mean |
| `z_score` | float | Deviation from personal baseline |
| `elevated_risk` | bool | True if alert should be triggered |

---

## Testing

**Script:** `test_pipeline.py`  
**Run:** `python test_pipeline.py`

### Purpose
End-to-end validation that walks through all **3 clinical stages** using synthetic sensor data, verifying that the pipeline correctly detects crisis behaviour after SSRI treatment begins.

### Test Flow

| Stage | Duration | Sensor Data | System Behaviour |
|---|---|---|---|
| **Stage 1: Training** | N/A | Uses pre-trained models from the real StudentLife dataset | Loads `models/*.pkl` — no new training needed |
| **Stage 2: Baseline** | 24 days | Normal daily behaviour (all 9 sensor sources) | Builds personal baseline, **no alerts triggered** |
| **Stage 3: Post-SSRI** | 30 days | 10 normal + 10 crisis + 10 recovery | Alerts are **active** — detects deviations |

### Stage 3 Phases (Post-SSRI Detail)

| Phase | Days | Sensor Pattern | Expected Outcome |
|---|---|---|---|
| Normal | 1-10 | Healthy behaviour (same as baseline) | Minimal/no alerts |
| **Crisis** | **11-20** | **SSRI side effects** — insomnia, isolation, restlessness | **Elevated risk alerts triggered** |
| Recovery | 21-30 | Return to healthy behaviour | Alerts should stop |

### Crisis Sensor Patterns
During crisis days (simulating SSRI-induced side effects), the synthetic data generates:
- **Insomnia:** 1-3 hrs dark time vs 6-10 normal, 8-20 night unlocks vs 0-3 normal
- **Social isolation:** 0-2 conversations vs 3-15 normal, 3-10 missed calls vs 0-2 normal
- **Restless phone usage:** 100-200 unlocks vs 30-80 normal, short anxious sessions
- **Physical isolation:** 0-2 nearby Bluetooth devices vs 3-12 normal

### Expected Output
The test prints a day-by-day timeline for each stage, then a summary:
- **Stage 2:** 24 days of baseline building, no alerts
- **Stage 3:** Alerts should fire during the crisis phase (days 11-20) and subside during recovery (days 21-30)
- **Pass condition:** Crisis phase has elevated alerts, normal/recovery phases have minimal alerts

---

## Project File Structure

```
BaselineAlgo/
|
|-- dataset/                          # Raw data directory
|   |-- sensing/
|   |   |-- phonelock/                # Phone unlock/lock events
|   |   |-- dark/                     # Screen off periods
|   |   |-- gps/                      # GPS coordinates
|   |   |-- conversation/             # Face-to-face conversations
|   |   |-- phonecharge/              # Charging sessions
|   |   |-- wifi_location/            # WiFi-based location labels
|   |   |-- bluetooth/                # Nearby Bluetooth devices
|   |-- call_log/                     # Call history
|   |-- sms/                          # SMS messages
|
|-- weight_features/                  # Pre-computed labels & scoring
|   |-- sensor_features.py            # Step 1: Feature engineering script
|   |-- restlessness_scores.csv       # PCA ground truth: restlessness
|   |-- impulsivity_scores.csv        # PCA ground truth: impulsivity
|   |-- irritability_scores.csv       # PCA ground truth: irritability
|   |-- insomnia_scores.csv           # PCA ground truth: insomnia
|   |-- *_weights.csv                 # PCA component weights
|   |-- *_scoring.py                  # Scoring scripts
|
|-- models/                           # Trained model weights
|   |-- restlessness_rf.pkl           # Random Forest: sensors -> restlessness
|   |-- impulsivity_rf.pkl            # Random Forest: sensors -> impulsivity
|   |-- irritability_rf.pkl           # Random Forest: sensors -> irritability
|   |-- insomnia_rf.pkl               # Random Forest: sensors -> insomnia
|
|-- sensor_daily_features.csv         # Step 1 output: daily features
|-- daily_suicide_risk.csv            # Step 3 output: risk scores & alerts
|
|-- train_models.py                   # Step 2: Train 4 RF models
|-- predict_risk.py                   # Step 3: Batch risk scoring
|-- alert_system.py                   # Step 4: Real-time alert system
|-- test_pipeline.py                  # End-to-end test with synthetic data
|-- DOCUMENTATION.md                  # This file
```

---

## Configuration Parameters

| Parameter | Value | Location | Description |
|---|---|---|---|
| `ROLLING_WINDOW` | 14 | `predict_risk.py`, `alert_system.py` | Days used for personal baseline (rolling mean) |
| `Z_THRESHOLD` | 1.5 | `predict_risk.py`, `alert_system.py` | Z-score above which an elevated risk alert is triggered |
| `n_estimators` | 100 | `train_models.py` | Number of trees in each Random Forest model |
| `n_splits` | 5 | `train_models.py` | Number of folds for cross-validation |
| Anomaly blend | 50/50 | `alert_system.py` | Weight split: 50% model predictions + 50% sensor anomaly |

---

## How to Run the Full Pipeline

```bash
# Step 1: Extract sensor features from raw data
python weight_features/sensor_features.py

# Step 2: Train the 4 construct models
python train_models.py

# Step 3: Generate batch risk scores with deviation detection
python predict_risk.py

# Step 4: Run the real-time alert system demo
python alert_system.py

# Test: Validate with synthetic data
python test_pipeline.py
```

---

## Dependencies

- Python 3.12+
- pandas
- numpy
- scikit-learn
- joblib
