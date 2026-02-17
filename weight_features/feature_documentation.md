# Feature Documentation – EMA Questions, Manipulations & Weights (Psychiatrist-Reviewed)

All features are normalised to **[0, 1]** where **higher = more of the negative trait**.  
Weights are produced by PCA (first principal component) and sum to 1.0.  
Final score = (weighted sum of normalised features) × 100.

> **Note:** Feature selection was reviewed and approved by a psychiatrist for clinical relevancy.

---

## 1. Restlessness (3 features)

| # | Feature | Survey | EMA Question | Options | Manipulation |
|---|---------|--------|-------------|---------|--------------|
| 1 | stress | Stress | *"Right now, I am..."* | [1] A little stressed, [2] Definitely stressed, [3] Stressed out, [4] Feeling good, [5] Feeling great | **Semantic mapping:** 5→0.00, 4→0.15, 1→0.40, 2→0.70, 3→1.00 |
| 2 | anxiety | Behavior | *"In the past 15 minutes, I was anxious, easily upset."* | (Not at all) 1 2 3 4 5 (Extremely) | **As-is:** (val−1)/4 |
| 3 | not_calm | Behavior | *"In the past 15 minutes, I was calm, emotionally stable."* | (Not at all) 1 2 3 4 5 (Extremely) | **Inverted:** (5−val)/4 |

---

## 2. Impulsivity (2 features)

| # | Feature | Survey | EMA Question | Options | Manipulation |
|---|---------|--------|-------------|---------|--------------|
| 1 | not_dependable | Behavior | *"In the past 15 minutes, I was dependable, self-disciplined."* | (Not at all) 1 2 3 4 5 (Extremely) | **Inverted:** (5−val)/4 |
| 2 | critical | Behavior | *"In the past 15 minutes, I was critical, quarrelsome."* | (Not at all) 1 2 3 4 5 (Extremely) | **As-is:** (val−1)/4 |

---

## 3. Irritability (3 features)

| # | Feature | Survey | EMA Question | Options | Manipulation |
|---|---------|--------|-------------|---------|--------------|
| 1 | critical | Behavior | *"In the past 15 minutes, I was critical, quarrelsome."* | (Not at all) 1 2 3 4 5 (Extremely) | **As-is:** (val−1)/4 |
| 2 | not_calm | Behavior | *"In the past 15 minutes, I was calm, emotionally stable."* | (Not at all) 1 2 3 4 5 (Extremely) | **Inverted:** (5−val)/4 |
| 3 | not_reserved | Behavior | *"In the past 15 minutes, I was reserved, quiet."* | (Not at all) 1 2 3 4 5 (Extremely) | **Inverted:** (5−val)/4 |

---

## 4. Insomnia (3 features)

| # | Feature | Survey | EMA Question | Options | Manipulation |
|---|---------|--------|-------------|---------|--------------|
| 1 | poor_sleep_quality | Sleep | *"How would you rate your overall sleep last night?"* | [1] Very good, [2] Fairly good, [3] Fairly bad, [4] Very bad | **As-is:** (val−1)/3 |
| 2 | few_hours | Sleep | *"How many hours did you sleep last night?"* | [1] <3h, [2] 3.5h, … [19] 12h | **Inverted:** 1−(val−1)/18 |
| 3 | daytime_sleepiness | Sleep | *"How often did you have trouble staying awake yesterday while in class, eating meals or engaging in social activity?"* | [1] None, [2] Once, [3] Twice, [4] Three or more times | **As-is:** (val−1)/3 |

---

## Notes

- **Daily aggregation**: multiple responses per day → daily mean per feature.
- **Missing data**: days with any feature missing are dropped (no interpolation).
- **Weights**: PCA first principal component absolute loadings, normalised to sum to 1. Weights will be regenerated when the scoring scripts are re-run.
