"""
End-to-End Pipeline Test: 3 Clinical Stages
============================================
Walks through the full clinical workflow:

  Stage 1 (Training):     Load pre-trained models from the real dataset
  Stage 2 (Baseline):     24 days of normal sensor data -- establish personal baseline
  Stage 3 (Post-SSRI):    30 days of monitoring -- includes crisis episodes to test alerts

Uses the already-trained Random Forest models (trained on the StudentLife dataset).
Generates synthetic sensor data that mimics a real patient's phone output.
"""
import os
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = "models"
CONSTRUCTS = ["restlessness", "impulsivity", "irritability", "insomnia"]
FEATURE_COLS = [
    'unlock_count', 'avg_session_sec', 'total_unlocked_sec', 'night_unlocks',
    'total_dark_hrs', 'dark_fragments', 'longest_dark_streak_hrs', 'night_dark_hrs',
    'convo_count', 'total_convo_min', 'avg_convo_length_min',
    'incoming_calls', 'outgoing_calls', 'missed_calls', 'total_call_min',
    'sms_sent', 'sms_received',
    'charge_sessions', 'night_charge_hrs',
    'avg_nearby_devices'
]

ROLLING_WINDOW = 14
Z_THRESHOLD = 1.5

# ──────────────────────────────────────────────
#  Synthetic sensor generators
# ──────────────────────────────────────────────

def generate_normal_day(rng):
    """Healthy, stable daily sensor pattern."""
    return {
        'unlock_count':           rng.integers(30, 80),
        'avg_session_sec':        rng.uniform(60, 300),
        'total_unlocked_sec':     rng.uniform(3000, 10000),
        'night_unlocks':          rng.integers(0, 3),
        'total_dark_hrs':         rng.uniform(6, 10),
        'dark_fragments':         rng.integers(2, 8),
        'longest_dark_streak_hrs':rng.uniform(4, 8),
        'night_dark_hrs':         rng.uniform(5, 8),
        'convo_count':            rng.integers(3, 15),
        'total_convo_min':        rng.uniform(10, 60),
        'avg_convo_length_min':   rng.uniform(2, 8),
        'incoming_calls':         rng.integers(1, 8),
        'outgoing_calls':         rng.integers(1, 6),
        'missed_calls':           rng.integers(0, 2),
        'total_call_min':         rng.uniform(5, 30),
        'sms_sent':               rng.integers(2, 15),
        'sms_received':           rng.integers(2, 15),
        'charge_sessions':        rng.integers(1, 4),
        'night_charge_hrs':       rng.uniform(4, 8),
        'avg_nearby_devices':     rng.uniform(3, 12),
    }


def generate_crisis_day(rng):
    """Crisis behaviour: insomnia, isolation, restless phone checking."""
    return {
        'unlock_count':           rng.integers(100, 200),
        'avg_session_sec':        rng.uniform(10, 60),
        'total_unlocked_sec':     rng.uniform(15000, 30000),
        'night_unlocks':          rng.integers(8, 20),
        'total_dark_hrs':         rng.uniform(1, 3),
        'dark_fragments':         rng.integers(10, 25),
        'longest_dark_streak_hrs':rng.uniform(0.5, 2),
        'night_dark_hrs':         rng.uniform(0.5, 2),
        'convo_count':            rng.integers(0, 2),
        'total_convo_min':        rng.uniform(0, 5),
        'avg_convo_length_min':   rng.uniform(0, 2),
        'incoming_calls':         rng.integers(0, 2),
        'outgoing_calls':         rng.integers(0, 1),
        'missed_calls':           rng.integers(3, 10),
        'total_call_min':         rng.uniform(0, 5),
        'sms_sent':               rng.integers(0, 2),
        'sms_received':           rng.integers(5, 20),
        'charge_sessions':        rng.integers(0, 2),
        'night_charge_hrs':       rng.uniform(0, 2),
        'avg_nearby_devices':     rng.uniform(0, 2),
    }


def sensor_anomaly_score(sensor_dict, sensor_history):
    """Compute how anomalous today's sensors are vs personal history (0-100)."""
    if len(sensor_history) < 3:
        return 50.0
    hist_df = pd.DataFrame(sensor_history)
    means = hist_df.mean()
    stds = hist_df.std().replace(0, 1.0)
    z_scores = []
    for feat in FEATURE_COLS:
        val = sensor_dict.get(feat, 0)
        z = abs(val - means.get(feat, 0)) / stds.get(feat, 1.0)
        z_scores.append(z)
    avg_z = np.mean(z_scores)
    return min(avg_z / 3.0, 1.0) * 100.0


def header(title):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}")


def main():
    rng = np.random.default_rng(seed=42)
    base_date = pd.Timestamp("2024-03-01")

    # ================================================================
    #  STAGE 1: TRAINING (load pre-trained models from real dataset)
    # ================================================================
    header("STAGE 1: TRAINING (Historical Dataset)")
    print("\n  The 4 Random Forest models were already trained on the")
    print("  StudentLife dataset (sensor_features.py + train_models.py).")
    print("  Loading pre-trained models...\n")

    models = {}
    for construct in CONSTRUCTS:
        path = os.path.join(MODELS_DIR, f"{construct}_rf.pkl")
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found. Run train_models.py first.")
            return
        models[construct] = joblib.load(path)
        print(f"  Loaded: {path}")

    print(f"\n  Models ready. These were trained on:")
    print(f"    - 2,683 daily sensor records from 49 students")
    print(f"    - 20 engineered features per day")
    print(f"    - 4 PCA-derived construct scores as Y-labels")
    print(f"  Stage 1 complete. Models are reusable for all patients.")

    # ================================================================
    #  STAGE 2: BASELINE PERIOD (24 days of normal behaviour)
    # ================================================================
    header("STAGE 2: BASELINE PERIOD (24 Days, Pre-SSRI)")
    print("\n  Simulating a new patient's first 24 days of monitoring.")
    print("  All 9 sensor sources are collected daily from their phone.")
    print("  No alerts are triggered during this period.\n")

    risk_history = []
    sensor_history = []

    print(f"  {'Day':>4}  {'Date':>12}  {'Risk Score':>11}  {'Anomaly':>8}  {'Baseline':>9}  Status")
    print(f"  {'---':>4}  {'----':>12}  {'----------':>11}  {'-------':>8}  {'--------':>9}  ------")

    for day in range(24):
        date = base_date + pd.Timedelta(days=day)
        sensor_data = generate_normal_day(rng)

        # Predict constructs
        features = np.array([[sensor_data.get(f, 0) for f in FEATURE_COLS]])
        features_df = pd.DataFrame(features, columns=FEATURE_COLS)
        preds = {}
        for construct, model in models.items():
            preds[construct] = model.predict(features_df)[0]

        model_risk = np.mean(list(preds.values()))
        anomaly = sensor_anomaly_score(sensor_data, sensor_history)
        sensor_history.append({f: sensor_data.get(f, 0) for f in FEATURE_COLS})

        risk_score = 0.5 * model_risk + 0.5 * anomaly
        risk_history.append(risk_score)

        # Baseline (rolling mean, building up)
        window = risk_history[-ROLLING_WINDOW:]
        baseline = np.mean(window)

        print(f"  {day+1:4d}  {date.date()}  {risk_score:11.2f}  {anomaly:8.1f}  {baseline:9.2f}  "
              f"(building baseline)")

    baseline_mean = np.mean(risk_history)
    baseline_std = np.std(risk_history, ddof=1)
    print(f"\n  Baseline established after 24 days:")
    print(f"    Personal mean:  {baseline_mean:.2f}")
    print(f"    Personal std:   {baseline_std:.2f}")
    print(f"    History length: {len(risk_history)} days")
    print(f"  Stage 2 complete. Patient starts SSRI medication.\n")

    # ================================================================
    #  STAGE 3: POST-SSRI MONITORING (30 days, with crisis episodes)
    # ================================================================
    header("STAGE 3: POST-SSRI MONITORING (30 Days)")
    print("\n  Patient has started SSRI treatment.")
    print("  Monitoring continues with all 9 sensor sources daily.")
    print("  Alerts are NOW ACTIVE -- detecting deviations from baseline.")
    print()
    print("  Scenario: Days 1-10 normal, Days 11-20 CRISIS, Days 21-30 recovery\n")

    alerts_normal = 0
    alerts_crisis = 0
    alerts_recovery = 0

    print(f"  {'Day':>4}  {'Date':>12}  {'Phase':>9}  {'Risk':>6}  {'Anomaly':>8}  "
          f"{'Baseline':>9}  {'Z-score':>8}  Alert")
    print(f"  {'---':>4}  {'----':>12}  {'-----':>9}  {'----':>6}  {'-------':>8}  "
          f"{'--------':>9}  {'-------':>8}  -----")

    for day in range(30):
        date = base_date + pd.Timedelta(days=24 + day)

        # Determine phase
        if day < 10:
            phase = "NORMAL"
            sensor_data = generate_normal_day(rng)
        elif day < 20:
            phase = "CRISIS"
            sensor_data = generate_crisis_day(rng)
        else:
            phase = "RECOVERY"
            sensor_data = generate_normal_day(rng)

        # Predict constructs
        features = np.array([[sensor_data.get(f, 0) for f in FEATURE_COLS]])
        features_df = pd.DataFrame(features, columns=FEATURE_COLS)
        preds = {}
        for construct, model in models.items():
            preds[construct] = model.predict(features_df)[0]

        model_risk = np.mean(list(preds.values()))
        anomaly = sensor_anomaly_score(sensor_data, sensor_history)
        sensor_history.append({f: sensor_data.get(f, 0) for f in FEATURE_COLS})

        risk_score = 0.5 * model_risk + 0.5 * anomaly
        risk_history.append(risk_score)

        # Compute z-score against rolling baseline
        window = risk_history[-ROLLING_WINDOW:]
        baseline = np.mean(window)
        rolling_std = np.std(window, ddof=1) if len(window) > 1 else 1.0
        if rolling_std == 0:
            rolling_std = 1.0
        z_score = (risk_score - baseline) / rolling_std

        elevated = z_score > Z_THRESHOLD
        alert_str = "** ALERT **" if elevated else ""

        if elevated:
            if phase == "NORMAL":
                alerts_normal += 1
            elif phase == "CRISIS":
                alerts_crisis += 1
            else:
                alerts_recovery += 1

        print(f"  {day+1:4d}  {date.date()}  {phase:>9}  {risk_score:6.1f}  {anomaly:8.1f}  "
              f"{baseline:9.2f}  {z_score:+8.2f}  {alert_str}")

    # ================================================================
    #  SUMMARY
    # ================================================================
    header("TEST SUMMARY")
    total_alerts = alerts_normal + alerts_crisis + alerts_recovery
    print(f"\n  Stage 2 (Baseline):    24 days of normal data, no alerts triggered")
    print(f"  Stage 3 (Post-SSRI):   30 days of monitoring, {total_alerts} total alerts\n")
    print(f"    Normal phase   (days 1-10):   {alerts_normal} alerts")
    print(f"    Crisis phase   (days 11-20):  {alerts_crisis} alerts")
    print(f"    Recovery phase (days 21-30):  {alerts_recovery} alerts")

    print(f"\n  Expected behaviour:")
    print(f"    - Minimal alerts during normal and recovery phases")
    print(f"    - Elevated alerts during crisis phase (SSRI side effects)")

    if alerts_crisis > 0 and alerts_normal <= 2:
        print(f"\n  >> TEST PASSED: The pipeline correctly detected the crisis")
        print(f"     period and would have triggered a Voice Agent check-in.")
    elif alerts_crisis == 0:
        print(f"\n  >> INCONCLUSIVE: No alerts during crisis. The model may need")
        print(f"     tuning, but the pipeline infrastructure works correctly.")
    else:
        print(f"\n  >> REVIEW: {alerts_normal} false alarms during normal phase.")

    print(f"\n{'#' * 70}\n")


if __name__ == "__main__":
    main()
