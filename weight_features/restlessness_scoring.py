"""
Restlessness Scoring (v4) – uses survey_preprocessing module.
Features: stress, anxiety, not_calm, disorganized.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from survey_preprocessing import (
    STUDENT_IDS, load_stress, load_anxiety, load_not_calm,
    load_disorganized, build_daily_features,
)

BASE_OUT = __import__("pathlib").Path(r"c:\Users\bzion\OneDrive - mail.tau.ac.il\CE\sem5\design_thinking")

FEATURE_COLS = ["stress_val", "anxiety_val", "not_calm_val", "disorganized_val"]
LOADERS = [load_stress, load_anxiety, load_not_calm, load_disorganized]


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
    full["restlessness_score"] = (full[FEATURE_COLS].values @ weights) * 100.0
    return full, weights, pca.explained_variance_ratio_[0], valid_sids


if __name__ == "__main__":
    print("=" * 60)
    print("RESTLESSNESS  v4  (4 features, semantic stress mapping)")
    print("=" * 60)

    full, weights, ev, sids = run_pca_scoring()
    print(f"\nStudents: {len(sids)}  |  Rows: {len(full)}  |  PC1: {ev:.1%}")
    print("\nWeights:")
    for c, w in sorted(zip(FEATURE_COLS, weights), key=lambda x: -x[1]):
        print(f"  {c:22s}  {w:.4f}")
    s = full["restlessness_score"]
    print(f"\nStats: mean={s.mean():.1f} median={s.median():.1f} std={s.std():.1f}")

    full.reset_index().rename(columns={"index": "date"}).to_csv(BASE_OUT / "restlessness_scores.csv", index=False)
    pd.DataFrame({"feature": FEATURE_COLS, "weight": weights}) \
      .sort_values("weight", ascending=False).reset_index(drop=True) \
      .to_csv(BASE_OUT / "restlessness_weights.csv", index=False)
    print("Saved restlessness_scores.csv + restlessness_weights.csv")
