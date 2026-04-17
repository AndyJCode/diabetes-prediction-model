import sys
import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocess import preprocess_data
from train import build_model, compute_data_version, log_config_params
from evaluation import evaluate_model

TRACKING_URL = f"sqlite:///{Path(__file__).parent / 'mlflow.db'}"
EXPERIMENT_NAME = "diabetes_classification"
BASE_CONFIG_PATH = 'configs/config.yaml'


def load_base_config(path: str = BASE_CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_experiment_variants(base_config: Dict) -> List[Dict]:
    """Build one config dict per model type, inheriting shared base settings."""
    model_variants = [
        
        {'model_type': 'random_forest', 
         'rf_n_estimators': 100, 'rf_max_depth': None, 'rf_bootstrap': True,
         'class_weight': None, 'random_state': 12345},

        {'model_type': 'gradient_boosting',
         'gb_n_estimators': 100, 'gb_learning_rate': 0.1, 'gb_max_depth': 3,
         'random_state': 12345},

        {'model_type': 'logistic_regression',
         'lr_C': 1.0, 'lr_max_iter': 1000,
         'class_weight': None, 'random_state': 12345},

        {'model_type': 'neural_network',
         'nn_hidden_layer_sizes': (100,), 'nn_learning_rate_init': 0.001,
         'nn_max_iter': 200, 'random_state': 12345},

        {'model_type': 'xgboost',
         'xgb_n_estimators': 100, 'xgb_learning_rate': 0.1, 'xgb_max_depth': 3,
         'eval_metric': 'logloss', 'random_state': 12345},

        {'model_type': 'lightgbm',
         'lgbm_n_estimators': 100, 'lgbm_learning_rate': 0.1, 'lgbm_max_depth': 3,
         'random_state': 12345},
    ]
    return [{**base_config, **variant} for variant in model_variants]


def run_experiment_from_dict(config: Dict, X_train, X_test, y_train, y_test,
                             data_version: str) -> str:
    """Train one model from a config dict and log results to MLflow."""
    model = build_model(config)
    model_type = config.get('model_type', 'unknown')

    with mlflow.start_run(run_name=model_type) as run:
        log_config_params(config)
        mlflow.log_param('data_version', data_version)

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, artifact_path='model')
        return run.info.run_id


def run_all_experiments(configs: List[Dict]) -> List[str]:
    """Preprocess data once, then train every model variant and log to MLflow."""
    mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = configs[0].get('data_path', 'data/diabetes.csv')
    X_train, X_test, y_train, y_test, _ = preprocess_data(data_path)
    data_version = compute_data_version(data_path)

    run_ids = []
    for i, config in enumerate(configs, start=1):
        print(f"\n=== Running experiment {i}/{len(configs)}: {config['model_type']} ===")
        run_id = run_experiment_from_dict(
            config, X_train, X_test, y_train, y_test, data_version
        )
        run_ids.append(run_id)
    return run_ids


def compare_experiments(primary_metric: str = "f1_score") -> None:
    """Retrieve completed MLflow runs, print a comparison table, and show the best."""
    mlflow.set_tracking_uri(TRACKING_URL)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found in tracking store.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{primary_metric} DESC"],
        max_results=50,
    )
    if runs.empty:
        print("No completed runs found for the experiment.")
        return

    print(f"\nTop 5 Runs by {primary_metric}:")
    print("=" * 80)
    for _, row in runs.head(5).iterrows():
        print(f"\n  Run:      {row['run_id'][:8]}...")
        print(f"  Model:    {row['params.model_type']}")
        print(f"  F1:       {row['metrics.f1_score']:.4f}")
        print(f"  Accuracy: {row['metrics.accuracy']:.4f}")
        print(f"  ROC-AUC:  {row['metrics.roc_auc']:.4f}")

    best = runs.iloc[0]
    print(f"\n{'=' * 80}")
    print("BEST MODEL")
    print(f"{'=' * 80}")
    print(f"  Run ID:    {best['run_id']}")
    print(f"  Model:     {best['params.model_type']}")
    print(f"  F1:        {best['metrics.f1_score']:.4f}")
    print(f"  Accuracy:  {best['metrics.accuracy']:.4f}")
    print(f"  ROC-AUC:   {best['metrics.roc_auc']:.4f}")

    print(f"\n{'=' * 80}")
    print("Average F1 Score by Model Type:")
    print(f"{'=' * 80}")
    summary = runs.groupby("params.model_type")["metrics.f1_score"].agg(["mean", "max", "count"])
    summary.columns = ["avg_f1", "best_f1", "num_runs"]
    print(summary.sort_values("best_f1", ascending=False).to_string())

def save_best_model(run_id: str, output_path: str) -> None:
    """Download the best model artifact from MLflow and save it locally."""
    mlflow.set_tracking_uri(TRACKING_URL)
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "model")
    model_files = list(Path(local_path).glob("*.pkl"))
    if not model_files:
        raise RuntimeError(f"No model file found in artifacts for run {run_id}.")
    model_file = model_files[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_file.rename(output_path)
    print(f"Best model saved to: {output_path}")


if __name__ == "__main__":
    base_config = load_base_config()
    configs = build_experiment_variants(base_config)
    run_all_experiments(configs)
    compare_experiments(primary_metric="f1_score")
    best_run_id = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id], 
                                     order_by=["metrics.f1_score DESC"], max_results=1).iloc[0]["run_id"]