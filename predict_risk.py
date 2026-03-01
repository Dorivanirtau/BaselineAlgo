import os
import pandas as pd
import numpy as np
import joblib

# Configuration
SENSOR_FEATURES_FILE = "sensor_daily_features.csv"
MODELS_DIR = "models"
CONSTRUCTS = ["restlessness", "impulsivity", "irritability", "insomnia"]
OUTPUT_FILE = "daily_suicide_risk.csv"
ROLLING_WINDOW = 14     # days for personal baseline
Z_THRESHOLD = 1.5       # z-score threshold for elevated risk flag

def separator(title):
    print(f"\n{'='*60}")
    print(f"  STEP: {title}")
    print(f"{'='*60}")

def main():
    # ── Step 1: Load sensor features ──
    separator("Load Sensor Features")
    df = pd.read_csv(SENSOR_FEATURES_FILE)
    df['date'] = pd.to_datetime(df['date'])
    feature_cols = [c for c in df.columns if c not in ['uid', 'date']]
    X = df[feature_cols].fillna(0)

    print(f"  Records loaded:  {df.shape[0]}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Unique students: {df['uid'].nunique()}")
    print(f"  Date range:      {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Feature columns: {feature_cols}")
    print(f"\n  Sample sensor data (first 5 rows):")
    print(df.head().to_string(index=False))

    # ── Step 2: Predict daily construct scores ──
    separator("Predict Daily Construct Scores")
    for construct in CONSTRUCTS:
        model_path = os.path.join(MODELS_DIR, f"{construct}_rf.pkl")
        if not os.path.exists(model_path):
            print(f"  ⚠ Model {model_path} not found. Setting {construct} predictions to 0.")
            df[f"pred_{construct}"] = 0.0
            continue

        model = joblib.load(model_path)
        df[f"pred_{construct}"] = model.predict(X)
        preds = df[f"pred_{construct}"]
        print(f"\n  {construct.upper()}:")
        print(f"    Model loaded from: {model_path}")
        print(f"    Prediction stats:  mean={preds.mean():.2f}, std={preds.std():.2f}, "
              f"min={preds.min():.2f}, max={preds.max():.2f}")

    pred_cols = [f"pred_{c}" for c in CONSTRUCTS]
    print(f"\n  Sample predictions (first 5 rows):")
    print(df[['uid', 'date'] + pred_cols].head().to_string(index=False))

    # ── Step 3: Composite suicide risk score ──
    separator("Compute Composite Suicide Risk Score")
    df['suicide_risk_score'] = df[pred_cols].mean(axis=1)
    srs = df['suicide_risk_score']
    print(f"  Formula: mean(pred_restlessness, pred_impulsivity, pred_irritability, pred_insomnia)")
    print(f"  Stats:   mean={srs.mean():.2f}, std={srs.std():.2f}, "
          f"min={srs.min():.2f}, max={srs.max():.2f}")
    print(f"\n  Sample composite scores (first 5 rows):")
    print(df[['uid', 'date', 'suicide_risk_score']].head().to_string(index=False))

    # ── Step 4: Personal baseline (rolling 14-day mean) ──
    separator(f"Compute Personal Baseline (Rolling {ROLLING_WINDOW}-day Mean)")
    df = df.sort_values(['uid', 'date'])
    df['baseline'] = df.groupby('uid')['suicide_risk_score'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    print(f"  Rolling window:  {ROLLING_WINDOW} days")
    print(f"  Baseline stats:  mean={df['baseline'].mean():.2f}, std={df['baseline'].std():.2f}")
    print(f"\n  Sample baseline vs. score (first 10 rows):")
    print(df[['uid', 'date', 'suicide_risk_score', 'baseline']].head(10).to_string(index=False))

    # ── Step 5: Deviation detection (z-score) ──
    separator("Deviation Detection (Z-Score)")
    df['rolling_std'] = df.groupby('uid')['suicide_risk_score'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=2).std()
    )
    # Fill NaN std (e.g., first days) with the student's overall std
    df['rolling_std'] = df.groupby('uid')['rolling_std'].transform(
        lambda x: x.fillna(x.mean())
    )
    # Avoid division by zero
    df['rolling_std'] = df['rolling_std'].replace(0, np.nan)
    df['rolling_std'] = df['rolling_std'].fillna(1.0)

    df['z_score'] = (df['suicide_risk_score'] - df['baseline']) / df['rolling_std']
    df['elevated_risk'] = (df['z_score'] > Z_THRESHOLD).astype(int)

    print(f"  Z-score formula:  (suicide_risk_score - baseline) / rolling_std")
    print(f"  Threshold:        z > {Z_THRESHOLD} → elevated_risk = 1")
    print(f"  Z-score stats:    mean={df['z_score'].mean():.2f}, std={df['z_score'].std():.2f}, "
          f"min={df['z_score'].min():.2f}, max={df['z_score'].max():.2f}")
    print(f"\n  Sample deviation analysis (first 10 rows):")
    print(df[['uid', 'date', 'suicide_risk_score', 'baseline', 'z_score', 'elevated_risk']].head(10).to_string(index=False))

    # ── Step 6: Save output ──
    separator("Save Output")
    output_cols = ['uid', 'date'] + pred_cols + [
        'suicide_risk_score', 'baseline', 'z_score', 'elevated_risk'
    ]
    df_out = df[output_cols].copy()
    df_out.to_csv(OUTPUT_FILE, index=False)

    # Summary stats
    n_elevated = df_out['elevated_risk'].sum()
    n_total = len(df_out)
    n_students = df_out['uid'].nunique()
    print(f"  Output file:       {OUTPUT_FILE}")
    print(f"  Output shape:      {df_out.shape}")
    print(f"  Total records:     {n_total}")
    print(f"  Unique students:   {n_students}")
    print(f"  Elevated risk days: {n_elevated} ({100*n_elevated/n_total:.1f}%)")

    # Show some elevated risk examples
    elevated = df_out[df_out['elevated_risk'] == 1]
    if not elevated.empty:
        print(f"\n  Sample ELEVATED RISK days:")
        print(elevated.head(10).to_string(index=False))
    else:
        print(f"\n  No elevated risk days detected.")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
