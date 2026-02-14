"""
Survey Preprocessing – all EMA loading & normalisation in one place.
====================================================================
Each loader returns a DataFrame with columns [date, <feature_name>],
where the feature is normalised to [0, 1] (higher = more of the trait).

Import this module in any scoring / modelling script:
    from survey_preprocessing import load_stress, load_anxiety, ...

Manipulation notes
------------------
Stress  – non-linear semantic mapping (see load_stress docstring).
Behavior – per-question linear scaling; inverted where needed.
Sleep   – per-question linear scaling; inverted where needed.
Study Spaces – productivity inverted to distraction.
Exercise     – walk as-is.
"""

import json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\bzion\OneDrive - mail.tau.ac.il\CE\sem5\design_thinking")
EMA  = BASE / "archive" / "dataset" / "EMA" / "response"

STUDENT_IDS = [
    0,1,2,3,4,5,7,8,9,10,12,13,14,15,16,17,18,19,20,22,23,24,25,27,
    30,31,32,33,34,35,36,39,41,42,43,44,45,46,47,49,50,51,52,53,54,56,57,58,59
]


# ── helpers ──────────────────────────────────────────────────────────────
def _safe_float(val):
    """Convert value to float, return NaN on failure."""
    if val in (None, "null", ""):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _load_json(path):
    """Load a JSON file, return empty list if missing/empty."""
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if txt in ("", "[]"):
        return []
    return json.loads(txt)


# ═════════════════════════════════════════════════════════════════════════
# STRESS SURVEY
# ═════════════════════════════════════════════════════════════════════════
# Original scale (NON-LINEAR):
#   [1] A little stressed
#   [2] Definitely stressed
#   [3] Stressed out          ← maximum stress
#   [4] Feeling good
#   [5] Feeling great         ← minimum stress
#
# Problem: Options 1-3 are stress grades, 4-5 are positive states.
#          Linear inversion (5-val)/4 is wrong because it treats
#          "A little stressed" (1) as maximum.
#
# Solution: Semantic mapping that respects the actual meaning.

_STRESS_MAP = {
    5: 0.00,   # Feeling great  → no stress
    4: 0.15,   # Feeling good   → minimal stress
    1: 0.40,   # A little stressed → mild stress
    2: 0.70,   # Definitely stressed → high stress
    3: 1.00,   # Stressed out   → maximum stress
}

def load_stress(sid):
    """Load stress with semantic (non-linear) mapping.

    Original scale: [1]A little stressed, [2]Definitely stressed,
    [3]Stressed out, [4]Feeling good, [5]Feeling great.

    Mapped to: 5→0.0, 4→0.15, 1→0.4, 2→0.7, 3→1.0
    (higher = more stressed)
    """
    rows = []
    for e in _load_json(EMA / "Stress" / f"Stress_u{sid:02d}.json"):
        v = _safe_float(e.get("level"))
        if np.isnan(v) or v not in _STRESS_MAP:
            continue
        rows.append({
            "date": datetime.fromtimestamp(e["resp_time"]).date(),
            "stress_val": _STRESS_MAP[v],
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# BEHAVIOR SURVEY  ("In the past 15 minutes, I was...")
# ═════════════════════════════════════════════════════════════════════════
# All questions are 1-5 Likert scale.
# Linear normalisation: as-is → (val-1)/4, inverted → (5-val)/4.

def load_anxiety(sid):
    """Behavior → 'anxious': 1-5 (1=not at all, 5=extremely anxious).
    Normalisation: (val-1)/4  [as-is, higher = more anxious]
    """
    rows = []
    for e in _load_json(EMA / "Behavior" / f"Behavior_u{sid:02d}.json"):
        v = _safe_float(e.get("anxious"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "anxiety_val": (v - 1.0) / 4.0})
    return pd.DataFrame(rows)


def load_not_calm(sid):
    """Behavior → 'calm': 1-5 (1=not calm, 5=very calm).
    Normalisation: (5-val)/4  [inverted: not calm → 1]
    """
    rows = []
    for e in _load_json(EMA / "Behavior" / f"Behavior_u{sid:02d}.json"):
        v = _safe_float(e.get("calm"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "not_calm_val": (5.0 - v) / 4.0})
    return pd.DataFrame(rows)


def load_disorganized(sid):
    """Behavior → 'disorganized': 1-5 (1=not at all, 5=very disorganized).
    Normalisation: (val-1)/4  [as-is, higher = more disorganized]
    """
    rows = []
    for e in _load_json(EMA / "Behavior" / f"Behavior_u{sid:02d}.json"):
        v = _safe_float(e.get("disorganized"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "disorganized_val": (v - 1.0) / 4.0})
    return pd.DataFrame(rows)


def load_critical(sid):
    """Behavior → 'critical': 1-5 (1=not at all, 5=very critical/quarrelsome).
    Normalisation: (val-1)/4  [as-is, higher = more critical]
    """
    rows = []
    for e in _load_json(EMA / "Behavior" / f"Behavior_u{sid:02d}.json"):
        v = _safe_float(e.get("critical"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "critical_val": (v - 1.0) / 4.0})
    return pd.DataFrame(rows)


def load_not_dependable(sid):
    """Behavior → 'dependable': 1-5 (1=not dependable, 5=very dependable).
    Normalisation: (5-val)/4  [inverted: not dependable → 1]
    """
    rows = []
    for e in _load_json(EMA / "Behavior" / f"Behavior_u{sid:02d}.json"):
        v = _safe_float(e.get("dependable"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "not_dependable_val": (5.0 - v) / 4.0})
    return pd.DataFrame(rows)


def load_not_reserved(sid):
    """Behavior → 'reserved': 1-5 (1=not reserved, 5=very reserved/quiet).
    Normalisation: (5-val)/4  [inverted: not reserved → 1]
    """
    rows = []
    for e in _load_json(EMA / "Behavior" / f"Behavior_u{sid:02d}.json"):
        v = _safe_float(e.get("reserved"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "not_reserved_val": (5.0 - v) / 4.0})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# SLEEP SURVEY
# ═════════════════════════════════════════════════════════════════════════

def load_poor_sleep_quality(sid):
    """Sleep → 'rate': [1]Very good, [2]Fairly good, [3]Fairly bad, [4]Very bad.
    Normalisation: (val-1)/3  [as-is, higher = worse sleep]
    """
    rows = []
    for e in _load_json(EMA / "Sleep" / f"Sleep_u{sid:02d}.json"):
        v = _safe_float(e.get("rate"))
        if np.isnan(v) or v < 1 or v > 4: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "poor_sleep_quality": (v - 1.0) / 3.0})
    return pd.DataFrame(rows)


def load_few_hours(sid):
    """Sleep → 'hour': 1-19 index ([1]<3h, [2]3.5h, ... [19]12h).
    Normalisation: 1-(val-1)/18  [inverted: fewer hours → 1]
    """
    rows = []
    for e in _load_json(EMA / "Sleep" / f"Sleep_u{sid:02d}.json"):
        v = _safe_float(e.get("hour"))
        if np.isnan(v) or v < 1 or v > 19: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "few_hours": 1.0 - (v - 1.0) / 18.0})
    return pd.DataFrame(rows)


def load_daytime_sleepiness(sid):
    """Sleep → 'social': [1]None, [2]Once, [3]Twice, [4]Three or more times.
    Normalisation: (val-1)/3  [as-is, higher = more trouble staying awake]
    """
    rows = []
    for e in _load_json(EMA / "Sleep" / f"Sleep_u{sid:02d}.json"):
        v = _safe_float(e.get("social"))
        if np.isnan(v) or v < 1 or v > 4: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "daytime_sleepiness": (v - 1.0) / 3.0})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# STUDY SPACES SURVEY
# ═════════════════════════════════════════════════════════════════════════

def load_distraction(sid):
    """Study Spaces → 'productivity': 1-4 (1=distracted, 4=productive).
    Normalisation: (4-val)/3  [inverted: distracted → 1]
    """
    rows = []
    for e in _load_json(EMA / "Study Spaces" / f"Study Spaces_u{sid:02d}.json"):
        v = _safe_float(e.get("productivity"))
        if np.isnan(v) or v < 1 or v > 4: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "distraction_val": (4.0 - v) / 3.0})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# EXERCISE SURVEY
# ═════════════════════════════════════════════════════════════════════════

def load_pacing(sid):
    """Exercise → 'walk': 1-5 (1=no walking, 5=long walk).
    Normalisation: (val-1)/4  [as-is, higher = more pacing]
    """
    rows = []
    for e in _load_json(EMA / "Exercise" / f"Exercise_u{sid:02d}.json"):
        v = _safe_float(e.get("walk"))
        if np.isnan(v) or v < 1 or v > 5: continue
        rows.append({"date": datetime.fromtimestamp(e["resp_time"]).date(),
                      "pacing_val": (v - 1.0) / 4.0})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# GENERIC UTILITIES
# ═════════════════════════════════════════════════════════════════════════

def build_daily_features(sid, loaders, feature_cols):
    """Load & merge features by date. No interpolation – drop incomplete rows.

    Parameters
    ----------
    sid : int – student id
    loaders : list of callables – each returns DataFrame with [date, feature]
    feature_cols : list of str – expected column names (one per loader)

    Returns
    -------
    DataFrame indexed by date, with columns = feature_cols, no NaN.
    """
    parts = {}
    for loader, col in zip(loaders, feature_cols):
        df = loader(sid)
        if df.empty:
            continue
        daily = df.groupby("date")[[col]].mean()
        parts[col] = daily

    if len(parts) < len(feature_cols):
        return pd.DataFrame()

    merged = None
    for col in feature_cols:
        if col not in parts:
            return pd.DataFrame()
        merged = parts[col] if merged is None else merged.join(parts[col], how="outer")

    merged = merged.dropna()
    merged.sort_index(inplace=True)
    return merged
