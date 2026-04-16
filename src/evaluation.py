'''
Model evaluation utilities: compute metrics, display a comparison table,
and select the best model from a set of experiment results.
'''

from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


def evaluate_model(model, X_test, y_test) -> Dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }


def print_comparison_table(results: List[Tuple[str, str, object, Dict]]) -> None:
    """
    Print a formatted comparison table across all models.

    Each entry in `results` is (run_id, model_type, model, metrics).
    """
    col_w = 22
    header = f"{'Model':<{col_w}}" + "".join(f"{m:>12}" for m in METRIC_KEYS)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for _, model_type, _, metrics in results:
        row = f"{model_type:<{col_w}}"
        for m in METRIC_KEYS:
            val = metrics.get(m)
            row += f"{val:>12.4f}" if val is not None else f"{'N/A':>12}"
        print(row)
    print(sep)


def select_best_model(
    results: List[Tuple[str, str, object, Dict]],
    selection_metric: str = "roc_auc",
) -> Optional[Tuple[str, str, object, Dict]]:
    """
    Return the (run_id, model_type, model, metrics) entry with the highest
    value for `selection_metric`. Returns None if no result has that metric.
    """
    valid = [r for r in results if r[3].get(selection_metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r[3][selection_metric])
