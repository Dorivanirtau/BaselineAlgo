"""
Insomnia Scoring (v2) – uses survey_preprocessing module.
Features: poor_sleep_quality, few_hours, daytime_sleepiness (all Sleep).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from survey_preprocessing import (
    STUDENT_IDS, load_poor_sleep_quality, load_few_hours,
    load_daytime_sleepiness, build_daily_features,
)

BASE_OUT = __import__("pathlib").Path(r"c:\Users\bzion\OneDrive - mail.tau.ac.il\CE\sem5\design_thinking")

FEATURE_COLS = ["poor_sleep_quality", "few_hours", "daytime_sleepiness"]
LOADERS = [load_poor_sleep_quality, load_few_hours, load_daytime_sleepiness]


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
    full["insomnia_score"] = (full[FEATURE_COLS].values @ weights) * 100.0
    return full, weights, pca.explained_variance_ratio_[0], valid_sids


if __name__ == "__main__":
    print("=" * 60)
    print("INSOMNIA  v2  (3 Sleep features)")
    print("=" * 60)

    full, weights, ev, sids = run_pca_scoring()
    print(f"\nStudents: {len(sids)}  |  Rows: {len(full)}  |  PC1: {ev:.1%}")
    print("\nWeights:")
    for c, w in sorted(zip(FEATURE_COLS, weights), key=lambda x: -x[1]):
        print(f"  {c:22s}  {w:.4f}")
    s = full["insomnia_score"]
    print(f"\nStats: mean={s.mean():.1f} median={s.median():.1f} std={s.std():.1f}")

    full.reset_index().rename(columns={"index": "date"}).to_csv(BASE_OUT / "insomnia_scores.csv", index=False)
    pd.DataFrame({"feature": FEATURE_COLS, "weight": weights}) \
      .sort_values("weight", ascending=False).reset_index(drop=True) \
      .to_csv(BASE_OUT / "insomnia_weights.csv", index=False)
    print("Saved insomnia_scores.csv + insomnia_weights.csv")
