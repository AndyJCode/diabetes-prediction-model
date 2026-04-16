import json
import os
import sys
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration
DRIFT_SHARE_WARNING  = 0.20   # warn if >20% of features drift
DRIFT_SHARE_CRITICAL = 0.40   # fail if >40% of features drift
P_VALUE_THRESHOLD    = 0.05   # K-S / Z-test significance level


def check_drift(reference_path, current_path):
    """Run drift analysis and return a structured result dict."""
    reference = pd.read_csv(reference_path)
    current   = pd.read_csv(current_path)

    report   = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    result   = snapshot.dict()

    metrics = result["metrics"]

    # ── Overall drift summary (first metric) ─────────────────────
    summary      = metrics[0]["value"]          # {'count': N, 'share': S}
    drifted_count = int(summary["count"])
    drift_share   = float(summary["share"])
    total_columns = len(metrics) - 1            # remaining metrics = per-column

    # ── Per-column p-values (metrics[1:]) ────────────────────────
    column_results = {}
    for m in metrics[1:]:
        name    = m["metric_name"]              # e.g. "ValueDrift(column=Glucose,...)"
        p_value = float(m["value"])
        # extract bare column name from metric_name
        col = name.split("column=")[1].split(",")[0]
        column_results[col] = {
            "p_value":       round(p_value, 4),
            "drift_detected": p_value < P_VALUE_THRESHOLD,
        }

    drifted_features = [col for col, v in column_results.items() if v["drift_detected"]]

    # ── Status ───────────────────────────────────────────────────
    if drift_share >= DRIFT_SHARE_CRITICAL:
        status = "critical"
    elif drift_share >= DRIFT_SHARE_WARNING:
        status = "warning"
    else:
        status = "ok"

    return {
        "total_features":      total_columns,
        "drifted_features":    drifted_count,
        "drift_share":         round(drift_share, 3),
        "dataset_drift":       drift_share >= DRIFT_SHARE_WARNING,
        "status":              status,
        "drifted_feature_names": drifted_features,
        "column_results":      column_results,
        "report_snapshot":     snapshot,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect_drift.py <reference.csv> <current.csv>")
        sys.exit(1)

    reference_path = sys.argv[1]
    current_path   = sys.argv[2]

    print(f"Checking drift: {current_path} vs {reference_path}")
    print("=" * 60)

    try:
        result = check_drift(reference_path, current_path)

        print(f"Features drifted: {result['drifted_features']}/{result['total_features']} "
              f"({result['drift_share']*100:.1f}%)")
        print(f"Dataset drift:    {result['dataset_drift']}")
        print(f"Status:           {result['status'].upper()}")

        if result["drifted_feature_names"]:
            print(f"\nDrifted features: {', '.join(result['drifted_feature_names'])}")

        print("\nPer-feature p-values (K-S test, threshold=0.05):")
        print(f"  {'Feature':<30} {'p-value':>8}  {'Drift':>6}")
        print(f"  {'-'*47}")
        for col, v in result["column_results"].items():
            flag = "YES" if v["drift_detected"] else "no"
            print(f"  {col:<30} {v['p_value']:>8.4f}  {flag:>6}")

        os.makedirs("reports", exist_ok=True)

        report_path = os.path.join("reports", "drift_check_report.html")
        result["report_snapshot"].save_html(report_path)
        print(f"\nHTML report saved to {report_path}")

        exportable = {k: v for k, v in result.items() if k != "report_snapshot"}
        with open("reports/drift_check_result.json", "w") as f:
            json.dump(exportable, f, indent=2)
        print("JSON result  saved to reports/drift_check_result.json")

        if result["status"] == "critical":
            print(f"\nCRITICAL: {result['drift_share']*100:.1f}% of features drifted "
                  f"(threshold: {DRIFT_SHARE_CRITICAL*100:.0f}%). Retraining required.")
            sys.exit(1)
        elif result["status"] == "warning":
            print(f"\nWARNING: {result['drift_share']*100:.1f}% of features drifted "
                  f"(threshold: {DRIFT_SHARE_WARNING*100:.0f}%). Monitor closely.")
            sys.exit(0)
        else:
            print("\nAll clear. Feature distributions are stable.")
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
