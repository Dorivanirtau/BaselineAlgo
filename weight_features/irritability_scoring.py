"""
Irritability Scoring (v3) – uses survey_preprocessing module.
Features: critical, not_calm, not_reserved (psychiatrist-reviewed).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from survey_preprocessing import (
    STUDENT_IDS, load_critical, load_not_calm,
    load_not_reserved, build_daily_features,
)

BASE_OUT = __import__("pathlib").Path(r"c:\Users\bzion\OneDrive - mail.tau.ac.il\CE\sem5\design_thinking")

FEATURE_COLS = ["critical_val", "not_calm_val", "not_reserved_val"]
LOADERS = [load_critical, load_not_calm, load_not_reserved]


def run_pca_scoring():
    frames, valid_sids = [], []
    for sid in STUDENT_IDS:
        df = build_daily_features(sid, LOADERS, FEATURE_COLS)
        if df.empty: continue
        df["student"] = f"u{sid:02d}"
        frames.append(df)
        valid_sids.append(sid)
    if not frames:
        raise RuntimeError("No data.")
    full = pd.concat(frames)

    pca = PCA(n_components=1)
    pca.fit(full[FEATURE_COLS].values)
    loadings = pca.components_[0]
    if np.sum(loadings < 0) > np.sum(loadings > 0):
        loadings = -loadings
    weights = np.abs(loadings) / np.abs(loadings).sum()
    full["irritability_score"] = (full[FEATURE_COLS].values @ weights) * 100.0
    return full, weights, pca.explained_variance_ratio_[0], valid_sids


if __name__ == "__main__":
    print("=" * 60)
    print("IRRITABILITY  v3  (3 Behavior features, psychiatrist-reviewed)")
    print("=" * 60)

    full, weights, ev, sids = run_pca_scoring()
    print(f"\nStudents: {len(sids)}  |  Rows: {len(full)}  |  PC1: {ev:.1%}")
    print("\nWeights:")
    for c, w in sorted(zip(FEATURE_COLS, weights), key=lambda x: -x[1]):
        print(f"  {c:22s}  {w:.4f}")
    s = full["irritability_score"]
    print(f"\nStats: mean={s.mean():.1f} median={s.median():.1f} std={s.std():.1f}")

    full.reset_index().rename(columns={"index": "date"}).to_csv(BASE_OUT / "irritability_scores.csv", index=False)
    pd.DataFrame({"feature": FEATURE_COLS, "weight": weights}) \
      .sort_values("weight", ascending=False).reset_index(drop=True) \
      .to_csv(BASE_OUT / "irritability_weights.csv", index=False)
    print("Saved irritability_scores.csv + irritability_weights.csv")
