'''
Train all configured model types (random forest, gradient boosting, logistic
regression, neural network, XGBoost, LightGBM) on the diabetes dataset.

Each model is evaluated on accuracy, precision, recall, F1, and ROC-AUC.
Results are compared and the best model is selected and logged to MLflow.
'''

import hashlib
import json
import joblib
import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess import preprocess_data
from evaluation import evaluate_model, print_comparison_table, select_best_model


def load_config(config_path: str = 'configs/config.yaml') -> Dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config: Dict) -> object:
    """Return a model instance based on the model_type in config."""
    model_type = config.get("model_type")
    random_state = config.get("random_state", 12345)

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', None),
            random_state=random_state,
            bootstrap=config.get('rf_bootstrap', True),
            class_weight=config.get('class_weight', None)
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config.get('gb_n_estimators', 100),
            learning_rate=config.get('gb_learning_rate', 0.1),
            max_depth=config.get('gb_max_depth', 3),
            random_state=random_state
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            C=config.get('lr_C', 1.0),
            max_iter=config.get('lr_max_iter', 1000),
            random_state=random_state,
            class_weight=config.get('class_weight', None)
        )
    elif model_type == "neural_network":
        return MLPClassifier(
            hidden_layer_sizes=config.get('nn_hidden_layer_sizes', (100,)),
            learning_rate_init=config.get('nn_learning_rate_init', 0.001),
            max_iter=config.get('nn_max_iter', 200),
            random_state=random_state
        )
    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=config.get('xgb_n_estimators', 100),
            learning_rate=config.get('xgb_learning_rate', 0.1),
            max_depth=config.get('xgb_max_depth', 3),
            random_state=random_state,
            eval_metric='logloss'
        )
    elif model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=config.get('lgbm_n_estimators', 100),
            learning_rate=config.get('lgbm_learning_rate', 0.1),
            max_depth=config.get('lgbm_max_depth', -1),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def compute_data_version(data_path: str) -> str:
    """Compute a deterministic SHA-256 hash of the dataset file."""
    hash_obj = hashlib.sha256()
    with open(Path(data_path).resolve(), 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def log_config_params(config: Dict) -> None:
    """Log all config key-value pairs as MLflow parameters."""
    for key, value in config.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif value is None:
            value = 'None'
        mlflow.log_param(key, str(value))


def _run_single(
    model_cfg: Dict,
    shared_cfg: Dict,
    X_train, X_test, y_train, y_test,
    data_version: str,
) -> Tuple[str, object, Dict]:
    """Train one model, log to MLflow, return (run_id, model, metrics)."""
    cfg = {**shared_cfg, **model_cfg}   # model-level keys override shared keys
    model = build_model(cfg)
    model_type = cfg["model_type"]

    with mlflow.start_run(run_name=model_type) as run:
        log_config_params(cfg)
        mlflow.log_param("data_version", data_version)

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id, model, metrics


def run_all_experiments(config_path: str = 'configs/config.yaml') -> Optional[str]:
    """
    Train every model listed in the config, compare results, and return the
    run_id of the best model (ranked by `selection_metric`, default roc_auc).
    """
    config = load_config(config_path)
    data_path = config.get('data_path', 'data/diabetes.csv')
    selection_metric = config.get('selection_metric', 'roc_auc')
    model_configs: List[Dict] = config.get('models', [])

    if not model_configs:
        raise ValueError("No models defined under 'models' in the config file.")

    shared_cfg = {k: v for k, v in config.items() if k not in ('models', 'selection_metric')}

    X_train, X_test, y_train, y_test, scaler = preprocess_data(data_path)
    data_version = compute_data_version(data_path)

    db_path = Path(__file__).parent.parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(config.get('experiment_name', 'diabetes_classification'))

    results: List[Tuple[str, str, object, Dict]] = []

    print(f"\nRunning {len(model_configs)} model(s)...\n")
    for model_cfg in model_configs:
        model_type = model_cfg.get("model_type", "unknown")
        print(f"  Training: {model_type}")
        run_id, model, metrics = _run_single(
            model_cfg, shared_cfg, X_train, X_test, y_train, y_test, data_version
        )
        results.append((run_id, model_type, model, metrics))

    print_comparison_table(results)

    best = select_best_model(results, selection_metric)
    if best is None:
        print(f"\nWarning: no model has a valid '{selection_metric}' score.")
        return None

    best_run_id, best_type, best_model, best_metrics = best
    print(f"\nBest model : {best_type}")
    print(f"  {selection_metric} = {best_metrics[selection_metric]:.4f}")
    print(f"  MLflow run ID: {best_run_id}")

    mlflow.tracking.MlflowClient().set_tag(best_run_id, "best_model", "true")
    save_best_model(best_model, scaler)
    return best_run_id


def save_best_model(model, scaler, path: str = 'best_model.pkl') -> None:
    """
    Save the best model and its scaler together so the app can load them
    at inference time and apply the exact same scaling used during training.
    """
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"\nBest model saved to {path}")


def load_best_model(path: str = 'best_model.pkl') -> tuple:
    """Load the saved model and scaler. Returns (model, scaler)."""
    bundle = joblib.load(path)
    return bundle['model'], bundle['scaler']


if __name__ == '__main__':
    run_all_experiments('configs/config.yaml')
