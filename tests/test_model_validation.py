import os
import sys
import pytest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import preprocess_data
from train import build_model
from evaluation import evaluate_model

# Absolute path so tests work regardless of where pytest is run from
DATA_PATH = str(Path(__file__).parent.parent / 'data' / 'diabetes.csv')

ALL_MODELS = [
    'random_forest',
    'gradient_boosting',
    'logistic_regression',
    'neural_network',
    'xgboost',
    'lightgbm',
]


@pytest.fixture(scope="module")
def data():
    """Load and preprocess the dataset once for all tests."""

    return preprocess_data(DATA_PATH)


@pytest.mark.parametrize("model_type", ALL_MODELS)
def test_model_predict_returns_binary_labels(data, model_type):

    """Every model's predictions must be 0 or 1 and match the test set length."""

    X_train, X_test, y_train, y_test, _ = data
    model = build_model({'model_type': model_type, 'random_state': 12345})
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (len(y_test),)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize("model_type", ALL_MODELS)
def test_model_meets_minimum_accuracy(data, model_type):

    """Every model must reach at least 0.70 accuracy on the test set."""

    X_train, X_test, y_train, y_test, _ = data
    model = build_model({'model_type': model_type, 'random_state': 12345})
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    assert accuracy > 0.70, f"{model_type} accuracy {accuracy:.4f} is below the 0.70 threshold."


@pytest.mark.parametrize("model_type", ALL_MODELS)
def test_evaluate_model_returns_all_metrics(data, model_type):

    """evaluate_model must return all five metrics with values between 0 and 1."""

    X_train, X_test, y_train, y_test, _ = data
    model = build_model({'model_type': model_type, 'random_state': 12345})
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    assert set(metrics.keys()) == expected_keys
    for key, value in metrics.items():
        assert value is not None and 0.0 <= value <= 1.0, (
            f"{model_type} metric '{key}' has invalid value: {value}"
        )

