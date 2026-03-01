import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, r2_score
import joblib

# Configuration
SENSOR_FEATURES_FILE = "sensor_daily_features.csv"
CONSTRUCTS = ["restlessness", "impulsivity", "irritability", "insomnia"]
MODELS_DIR = "models"
SCORES_DIR = "weight_features"

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 1. Load the sensor daily features (X base)
    print("Loading sensor daily features...")
    df_sensors = pd.read_csv(SENSOR_FEATURES_FILE)
    
    # Feature columns are everything except uid and date
    feature_cols = [c for c in df_sensors.columns if c not in ['uid', 'date']]
    print(f"Loaded {df_sensors.shape[0]} daily sensor records with {len(feature_cols)} features.")

    # Scoring metrics
    scoring = {
        'mae': make_scorer(mean_absolute_error),
        'rmse': make_scorer(root_mean_squared_error),
        'r2': make_scorer(r2_score)
    }

    # 2. Iterate through each of the 4 constructs
    for construct in CONSTRUCTS:
        print(f"\n{'='*50}")
        print(f"Training model for: {construct.upper()}")
        print(f"{'='*50}")

        scores_file = os.path.join(SCORES_DIR, f"{construct}_scores.csv")
        if not os.path.exists(scores_file):
            print(f"Warning: Scores file {scores_file} not found. Skipping {construct}.")
            continue
            
        # Load Y labels
        df_y = pd.read_csv(scores_file)
        
        # Merge X and Y
        # sensor data has 'uid' and 'date'
        # score data has 'student' and 'date', and the target is '{construct}_score'
        
        # ensure matching types for merge
        df_sensors['date'] = pd.to_datetime(df_sensors['date'])
        df_y['date'] = pd.to_datetime(df_y['date'])
        
        df_merged = pd.merge(
            df_sensors, 
            df_y, 
            left_on=['uid', 'date'], 
            right_on=['student', 'date'], 
            how='inner'
        )
        
        target_col = f"{construct}_score"
        if target_col not in df_merged.columns:
            print(f"Error: Target column '{target_col}' not found in merged data. Skipping {construct}.")
            continue
            
        # Drop days where the target score is NaN
        df_merged = df_merged.dropna(subset=[target_col])
        print(f"Merged dataset shape: {df_merged.shape}")
        
        if df_merged.empty:
            print(f"No overlapping data found for {construct}. Skipping.")
            continue

        # Extract X and y arrays
        X = df_merged[feature_cols].copy()
        
        # We fill any remaining NaNs in X with globally 0 just in case sensor features had missing columns globally
        X = X.fillna(0)
        y = df_merged[target_col]

        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape:   {y.shape}")

        # Initialize Model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Evaluate with 5-Fold Cross Validation
        print("Running 5-Fold Cross Validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring)
        
        avg_mae = np.mean(cv_results['test_mae'])
        avg_rmse = np.mean(cv_results['test_rmse'])
        avg_r2 = np.mean(cv_results['test_r2'])
        
        print(f"Results for {construct.upper()}:")
        print(f"  MAE:  {avg_mae:.4f}")
        print(f"  RMSE: {avg_rmse:.4f}")
        print(f"  R2:   {avg_r2:.4f}")

        # Train on the entire dataset for the final model
        print("Training final model on full dataset...")
        model.fit(X, y)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{construct}_rf.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
