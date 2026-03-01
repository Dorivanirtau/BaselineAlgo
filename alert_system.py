"""
Alert System (Step 4)
=====================
A real-time alert module that monitors students day-by-day.
For each new day of sensor data, it:
  1. Predicts the 4 construct scores using trained RF models
  2. Computes a composite suicide_risk_score
  3. Compares it against the student's personal rolling 14-day baseline
  4. Triggers an ALERT if the z-score exceeds the threshold
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


class StudentMonitor:
    """Monitors a single student's daily risk over time."""

    def __init__(self, uid, models):
        self.uid = uid
        self.models = models  # dict: construct_name -> trained model
        self.history = []     # list of daily composite risk scores
        self.sensor_history = []  # list of dicts (raw sensor readings per day)

    def _sensor_anomaly_score(self, sensor_dict):
        """
        Compute how anomalous today's sensors are compared to the student's
        own historical sensor patterns. Returns a 0-100 score where higher
        means more anomalous.
        """
        if len(self.sensor_history) < 3:
            return 50.0  # neutral when not enough history

        hist_df = pd.DataFrame(self.sensor_history)
        means = hist_df.mean()
        stds = hist_df.std().replace(0, 1.0)  # avoid div by zero

        # Compute z-score for each sensor feature
        z_scores = []
        for feat in FEATURE_COLS:
            val = sensor_dict.get(feat, 0)
            z = abs(val - means.get(feat, 0)) / stds.get(feat, 1.0)
            z_scores.append(z)

        # Average absolute z-score across all features, scaled to 0-100
        avg_z = np.mean(z_scores)
        # Normalize: z=0 -> 0, z=3 -> 100 (capped)
        anomaly = min(avg_z / 3.0, 1.0) * 100.0
        return anomaly

    def process_new_day(self, date, sensor_dict):
        """
        Process one new day of sensor data.

        Args:
            date: the date string or datetime
            sensor_dict: dict with keys matching FEATURE_COLS

        Returns:
            dict with prediction details and alert status
        """
        # Build feature vector
        features = np.array([[sensor_dict.get(f, 0) for f in FEATURE_COLS]])
        features_df = pd.DataFrame(features, columns=FEATURE_COLS)

        # Predict each construct
        predictions = {}
        for construct, model in self.models.items():
            pred = model.predict(features_df)[0]
            predictions[f"pred_{construct}"] = pred

        # Model-based risk score (average of 4 construct predictions)
        model_risk = np.mean(list(predictions.values()))

        # Sensor anomaly score (how unusual are today's raw sensors vs history)
        sensor_anomaly = self._sensor_anomaly_score(sensor_dict)

        # Store raw sensors for future anomaly calculations
        self.sensor_history.append({f: sensor_dict.get(f, 0) for f in FEATURE_COLS})

        # Composite risk score: blend model predictions (50%) + sensor anomaly (50%)
        risk_score = 0.5 * model_risk + 0.5 * sensor_anomaly

        # Add to history
        self.history.append(risk_score)

        # Compute baseline & z-score
        if len(self.history) >= 2:
            window = self.history[-ROLLING_WINDOW:]
            baseline = np.mean(window)
            rolling_std = np.std(window, ddof=1) if len(window) > 1 else 1.0
            if rolling_std == 0:
                rolling_std = 1.0
            z_score = (risk_score - baseline) / rolling_std
        else:
            baseline = risk_score
            z_score = 0.0

        elevated = z_score > Z_THRESHOLD

        result = {
            'uid': self.uid,
            'date': str(date),
            **predictions,
            'model_risk': model_risk,
            'sensor_anomaly': sensor_anomaly,
            'suicide_risk_score': risk_score,
            'baseline': baseline,
            'z_score': z_score,
            'elevated_risk': elevated,
            'history_length': len(self.history)
        }

        return result


def load_models():
    """Load all 4 trained RF models."""
    models = {}
    for construct in CONSTRUCTS:
        path = os.path.join(MODELS_DIR, f"{construct}_rf.pkl")
        if os.path.exists(path):
            models[construct] = joblib.load(path)
            print(f"  Loaded model: {construct}")
        else:
            print(f"  WARNING: Model not found: {path}")
    return models


def generate_normal_day(rng):
    return {
        'unlock_count': rng.integers(30, 80),
        'avg_session_sec': rng.uniform(60, 300),
        'total_unlocked_sec': rng.uniform(3000, 10000),
        'night_unlocks': rng.integers(0, 3),
        'total_dark_hrs': rng.uniform(6, 10),
        'dark_fragments': rng.integers(2, 8),
        'longest_dark_streak_hrs': rng.uniform(4, 8),
        'night_dark_hrs': rng.uniform(5, 8),
        'convo_count': rng.integers(3, 15),
        'total_convo_min': rng.uniform(10, 60),
        'avg_convo_length_min': rng.uniform(2, 8),
        'incoming_calls': rng.integers(1, 8),
        'outgoing_calls': rng.integers(1, 6),
        'missed_calls': rng.integers(0, 2),
        'total_call_min': rng.uniform(5, 30),
        'sms_sent': rng.integers(2, 15),
        'sms_received': rng.integers(2, 15),
        'charge_sessions': rng.integers(1, 4),
        'night_charge_hrs': rng.uniform(4, 8),
        'avg_nearby_devices': rng.uniform(3, 12),
    }


def generate_crisis_day(rng):
    return {
        'unlock_count': rng.integers(100, 200),
        'avg_session_sec': rng.uniform(10, 60),
        'total_unlocked_sec': rng.uniform(15000, 30000),
        'night_unlocks': rng.integers(8, 20),
        'total_dark_hrs': rng.uniform(1, 3),
        'dark_fragments': rng.integers(10, 25),
        'longest_dark_streak_hrs': rng.uniform(0.5, 2),
        'night_dark_hrs': rng.uniform(0.5, 2),
        'convo_count': rng.integers(0, 2),
        'total_convo_min': rng.uniform(0, 5),
        'avg_convo_length_min': rng.uniform(0, 2),
        'incoming_calls': rng.integers(0, 2),
        'outgoing_calls': rng.integers(0, 1),
        'missed_calls': rng.integers(3, 10),
        'total_call_min': rng.uniform(0, 5),
        'sms_sent': rng.integers(0, 2),
        'sms_received': rng.integers(5, 20),
        'charge_sessions': rng.integers(0, 2),
        'night_charge_hrs': rng.uniform(0, 2),
        'avg_nearby_devices': rng.uniform(0, 2),
    }


def demo():
    """Run a live demo simulating day-by-day monitoring."""
    print("=" * 60)
    print("  ALERT SYSTEM DEMO")
    print("  Simulating real-time daily monitoring")
    print("=" * 60)

    print("\nLoading trained models...")
    models = load_models()

    rng = np.random.default_rng(seed=123)
    base_date = pd.Timestamp("2024-01-01")

    # ── Monitor: Stable Student ──
    print("\n" + "-" * 60)
    print("  STUDENT: stable_student (30 days, all normal)")
    print("-" * 60)
    stable_monitor = StudentMonitor("stable_student", models)
    stable_alerts = 0
    for day in range(30):
        date = base_date + pd.Timedelta(days=day)
        sensor_data = generate_normal_day(rng)
        result = stable_monitor.process_new_day(date, sensor_data)
        status = "** ALERT **" if result['elevated_risk'] else "  ok"
        if result['elevated_risk']:
            stable_alerts += 1
        print(f"  Day {day+1:2d} | {date.date()} | anomaly={result['sensor_anomaly']:5.1f} | risk={result['suicide_risk_score']:6.2f} | "
              f"baseline={result['baseline']:6.2f} | z={result['z_score']:+5.2f} | {status}")

    print(f"\n  Total alerts: {stable_alerts} / 30 days")

    # ── Monitor: Crisis Student ──
    print("\n" + "-" * 60)
    print("  STUDENT: crisis_student (14 normal + 7 crisis)")
    print("-" * 60)
    crisis_monitor = StudentMonitor("crisis_student", models)
    crisis_alerts_normal = 0
    crisis_alerts_crisis = 0
    for day in range(21):
        date = base_date + pd.Timedelta(days=day)
        if day < 14:
            sensor_data = generate_normal_day(rng)
            phase = "NORMAL"
        else:
            sensor_data = generate_crisis_day(rng)
            phase = "CRISIS"

        result = crisis_monitor.process_new_day(date, sensor_data)
        status = "** ALERT **" if result['elevated_risk'] else "  ok"
        if result['elevated_risk']:
            if day < 14:
                crisis_alerts_normal += 1
            else:
                crisis_alerts_crisis += 1

        print(f"  Day {day+1:2d} | {date.date()} | {phase:6s} | anomaly={result['sensor_anomaly']:5.1f} | risk={result['suicide_risk_score']:6.2f} | "
              f"baseline={result['baseline']:6.2f} | z={result['z_score']:+5.2f} | {status}")

    print(f"\n  Alerts during normal phase (days 1-14):  {crisis_alerts_normal}")
    print(f"  Alerts during crisis phase (days 15-21): {crisis_alerts_crisis}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  DEMO SUMMARY")
    print("=" * 60)
    print(f"  Stable student: {stable_alerts} alerts in 30 days")
    print(f"  Crisis student: {crisis_alerts_normal} alerts (normal) + {crisis_alerts_crisis} alerts (crisis)")

    if crisis_alerts_crisis > 0:
        print("\n  >> The alert system DETECTED the crisis period!")
        print("  >> In a real deployment, this would trigger a Voice Agent")
        print("     check-in call to the student.")
    else:
        print("\n  >> The alert system did not trigger during the crisis phase.")
        print("  >> Model tuning or additional features may be needed.")

    print("=" * 60)


if __name__ == "__main__":
    demo()
