# Feature Documentation – Survey Questions, Manipulations & Weights

This document describes, for each of the four modeled constructs, which EMA survey questions are used, how each question is normalised, and the PCA-derived weight assigned to it.

> **All features are normalised to [0, 1] where higher = more of the negative trait.**  
> Weights are produced by PCA (first principal component) and sum to 1.0.  
> Final score = (weighted sum of normalised features) × 100.

---

## 1. Restlessness

| # | Feature | Survey | JSON Field | Original Scale | Normalisation (Manipulation) | Weight |
|---|---------|--------|------------|----------------|------------------------------|--------|
| 1 | stress | Stress | `level` | 1–5 (non-linear: 1="A little stressed", 2="Definitely stressed", 3="Stressed out", 4="Feeling good", 5="Feeling great") | **Semantic mapping** — 5→0.00, 4→0.15, 1→0.40, 2→0.70, 3→1.00 (not a simple inversion) | 0.2171 |
| 2 | anxiety | Behavior | `anxious` | 1–5 Likert (1=not at all, 5=extremely anxious) | **As-is:** (val − 1) / 4 | 0.3564 |
| 3 | not_calm | Behavior | `calm` | 1–5 Likert (1=not calm, 5=very calm) | **Inverted:** (5 − val) / 4 | 0.2438 |
| 4 | disorganized | Behavior | `disorganized` | 1–5 Likert (1=not at all, 5=very disorganized) | **As-is:** (val − 1) / 4 | 0.1826 |

---

## 2. Impulsivity

| # | Feature | Survey | JSON Field | Original Scale | Normalisation (Manipulation) | Weight |
|---|---------|--------|------------|----------------|------------------------------|--------|
| 1 | not_dependable | Behavior | `dependable` | 1–5 Likert (1=not dependable, 5=very dependable) | **Inverted:** (5 − val) / 4 | 0.2287 |
| 2 | disorganized | Behavior | `disorganized` | 1–5 Likert (1=not at all, 5=very disorganized) | **As-is:** (val − 1) / 4 | 0.2395 |
| 3 | critical | Behavior | `critical` | 1–5 Likert (1=not at all, 5=very critical/quarrelsome) | **As-is:** (val − 1) / 4 | 0.2726 |
| 4 | not_calm | Behavior | `calm` | 1–5 Likert (1=not calm, 5=very calm) | **Inverted:** (5 − val) / 4 | 0.2592 |

---

## 3. Irritability

| # | Feature | Survey | JSON Field | Original Scale | Normalisation (Manipulation) | Weight |
|---|---------|--------|------------|----------------|------------------------------|--------|
| 1 | anxiety | Behavior | `anxious` | 1–5 Likert (1=not at all, 5=extremely anxious) | **As-is:** (val − 1) / 4 | 0.3355 |
| 2 | critical | Behavior | `critical` | 1–5 Likert (1=not at all, 5=very critical/quarrelsome) | **As-is:** (val − 1) / 4 | 0.3231 |
| 3 | not_calm | Behavior | `calm` | 1–5 Likert (1=not calm, 5=very calm) | **Inverted:** (5 − val) / 4 | 0.2996 |
| 4 | not_reserved | Behavior | `reserved` | 1–5 Likert (1=not reserved, 5=very reserved/quiet) | **Inverted:** (5 − val) / 4 | 0.0419 |

---

## 4. Insomnia

| # | Feature | Survey | JSON Field | Original Scale | Normalisation (Manipulation) | Weight |
|---|---------|--------|------------|----------------|------------------------------|--------|
| 1 | poor_sleep_quality | Sleep | `rate` | 1–4 (1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad) | **As-is:** (val − 1) / 3 | 0.5741 |
| 2 | few_hours | Sleep | `hour` | 1–19 index (1=<3h, 2=3.5h, … 19=12h) | **Inverted:** 1 − (val − 1) / 18 | 0.1283 |
| 3 | daytime_sleepiness | Sleep | `social` | 1–4 (1=None, 2=Once, 3=Twice, 4=Three or more times trouble staying awake) | **As-is:** (val − 1) / 3 | 0.2976 |

---

## Additional Notes

- **Daily aggregation**: when a student has multiple survey responses on the same day, the daily mean is used for each feature.
- **Missing data**: days with any feature missing are dropped entirely (no interpolation).
- **Weight derivation**: PCA is fitted on all students' data pooled together; the first principal component's absolute loadings are normalised to sum to 1 to produce the weights.  If majority of loadings are negative, they are sign-flipped so that higher score = more of the trait.
