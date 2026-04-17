# Drift Analysis Report

**Dataset:** Pima Indian Diabetes Dataset (`data/diabetes.csv`)  
**Tool:** Evidently v0.7.21  
**Test:** Kolmogorov-Smirnov (continuous features) / Z-test (binary target)  
**p-value threshold:** 0.05  
**Reference split:** first 70% of rows (537 samples)  
**Current split:** last 30% of rows (231 samples)

---

## Summary

No drift was detected in any feature. most p-values are above the 0.05 significance threshold, indicating that the reference and current splits come from the same underlying distribution from the same static data
---

## Per-Feature Results

Features drifted: 0/9 (0.0%)
Dataset drift:    False
Status:           OK

Per-feature p-values (K-S test, threshold=0.05):
  Feature                         p-value   Drift
  -----------------------------------------------
  Pregnancies                      0.9512      no
  Glucose                          0.7018      no
  BloodPressure                    0.1111      no
  SkinThickness                    0.7968      no
  Insulin                          0.2457      no
  BMI                              0.6247      no
  DiabetesPedigreeFunction         0.3619      no
  Age                              0.5065      no
  Outcome                          0.7905      no

The lowest p-value was **BloodPressure (0.1111)**, which is still comfortably above the threshold. All other features show high p-values (>0.24), confirming strong distributional consistency across splits.

---

## Interpretation

The absence of drift is consistent with how this dataset was collected — it is a curated, static snapshot from a single study (NIDDK), not a live production stream. In a real deployment scenario:

- **Reference data** would be the training set used when the model was originally deployed
- **Current data** would be incoming patient records over a monitoring window (e.g., last 30 days)
- Drift would signal population shift (e.g., demographic changes, updated measurement equipment, seasonal effects) that may degrade model performance

---

## Thresholds & Actions

| Status | Condition | Recommended Action |
|---|---|---|
| OK | < 20% features drifted | Continue monitoring on schedule |
| Warning | 20–39% features drifted | Increase monitoring frequency; investigate drifted features |
| Critical | ≥ 40% features drifted | Halt predictions; retrain model on updated data |

---

## Artifacts

- **HTML report:** `reports/drift_check_report.html`
- **JSON result:** `reports/drift_check_result.json`

---

## How to Re-run

```bash
python detect_drift.py data/reference.csv data/current.csv
```

In production, replace `data/current.csv` with a fresh export of recent incoming data to monitor for real distributional shift over time.

---

## Script Overview (`detect_drift.py`)

| Function | Purpose |
|---|---|
| `check_drift(reference_path, current_path)` | Loads both CSVs, runs Evidently `DataDriftPreset`, parses per-feature p-values and overall drift share |
| Status thresholds | OK < 20% drifted · Warning 20–39% · Critical ≥ 40% |
| Outputs | Console summary, `reports/drift_check_report.html`, `reports/drift_check_result.json` |
| Exit code | `0` = OK or Warning · `1` = Critical (suitable for CI/CD gating) |

**Note:** The script uses the Evidently v0.7+ API (`metrics[0]["value"]` for summary, `metrics[1+]["value"]` for per-column p-values). Earlier Evidently versions used a different dict structure (`result["metrics"][0]["result"]`) which would cause a `KeyError`.
