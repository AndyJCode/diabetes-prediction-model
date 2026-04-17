# Drift Analysis

**Dataset:** Pima Indian Diabetes Dataset (`data/diabetes.csv`)  
**Tool:** Evidently v0.7.21  
**Test used:** Kolmogorov-Smirnov for continuous features, Z-test for the binary target  
**p-value threshold:** 0.05  
**Reference split:** first 70% of rows (537 samples)  
**Current split:** last 30% of rows (231 samples)

---

## Summary

No drift was detected in any of the 9 features. All p-values are well above the 0.05 threshold, which tells us the reference and current splits come from the same underlying distribution. This is expected since both splits come from the same static dataset.

| Metric | Value |
|---|---|
| Features checked | 9 |
| Features with drift | 0 |
| Drift share | 0.0% |
| Overall status | **OK** |

---

## Per-Feature Results

| Feature | p-value | Drift |
|---|---|---|
| Pregnancies | 0.9512 | No |
| Glucose | 0.7018 | No |
| BloodPressure | 0.1111 | No |
| SkinThickness | 0.7968 | No |
| Insulin | 0.2457 | No |
| BMI | 0.6247 | No |
| DiabetesPedigreeFunction | 0.3619 | No |
| Age | 0.5065 | No |
| Outcome (target) | 0.7905 | No |

BloodPressure had the lowest p-value at 0.1111, but that's still comfortably above the 0.05 cutoff. Everything else came back with high p-values, so there's nothing concerning here.

---

## What This Means in Practice

No drift showed up. This dataset is a one-time snapshot from a single study, not a live data stream. Both splits were drawn from the same source so they look the same.

In a real deployment though, drift actually matters. Here's how it would work:

- **Reference data** — the training set from when the model was first deployed
- **Current data** — new incoming patient records over some monitoring window, like the last 30 days
- **Why it matters** — if the population shifts (different demographics, updated lab equipment, seasonal patterns), the model's predictions can degrade without any obvious error showing up

That's why monitoring for drift is worth building in even if it doesn't trigger on this dataset.

---

## Status Thresholds

| Status | Condition | What to Do |
|---|---|---|
| OK | Less than 20% of features drifted | Keep monitoring on schedule |
| Warning | 20–39% of features drifted | Look into which features are drifting and increase monitoring frequency |
| Critical | 40% or more features drifted | Stop making predictions and retrain on updated data |

---

## How to Re-run

```bash
python detect_drift.py data/reference.csv data/current.csv
```

In production, swap `data/current.csv` for a fresh export of recent incoming records. The script outputs a console summary, an HTML report at `reports/drift_check_report.html`, and a JSON result at `reports/drift_check_result.json`.

Exit code `0` means OK or Warning. Exit code `1` means Critical, which makes it easy to gate on in a CI/CD pipeline.
