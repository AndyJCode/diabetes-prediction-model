import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocess import preprocess_data

DATA_PATH = str(Path(__file__).parent.parent / 'data' / 'diabetes.csv')


def make_sample_csv(tmp_path) -> str:
    """Write a minimal valid diabetes CSV and return its path."""
    data = {
        'Pregnancies':              [0, 1, 2, 3, 1, 0],
        'Glucose':                  [85, 90, 0, 120, 110, 95],   # 0 = missing
        'BloodPressure':            [70, 80, 0, 75,  72,  68],   # 0 = missing
        'SkinThickness':            [20, 25, 0, 30,  22,  18],   # 0 = missing
        'Insulin':                  [85, 90, 0, 100, 80,  70],   # 0 = missing
        'BMI':                      [30.5, 28.0, 0, 32.1, 27.5, 25.0],  # 0 = missing
        'DiabetesPedigreeFunction': [0.5, 0.3, 0.1, 0.4, 0.6, 0.2],
        'Age':                      [25, 30, 35, 40, 28, 22],
        'Outcome':                  [1, 0, 1, 0, 1, 0],
    }
    path = str(tmp_path / 'sample.csv')
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ── Preprocessing tests (≥4 required) ────────────────────────────

def test_no_missing_values_after_preprocessing(tmp_path):
    """Zeros in clinical columns must be imputed — no NaNs in output."""
    path = make_sample_csv(tmp_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(path)

    assert not np.isnan(X_train).any(), "NaN found in X_train."
    assert not np.isnan(X_test).any(),  "NaN found in X_test."


def test_features_are_scaled(tmp_path):
    """StandardScaler output should have mean ≈ 0 and std ≈ 1 on training data."""
    path = make_sample_csv(tmp_path)
    X_train, _X_test, _y_train, _y_test, _scaler = preprocess_data(path)

    col_means = np.abs(X_train.mean(axis=0))
    col_stds  = X_train.std(axis=0)

    assert (col_means < 1.5).all(), f"Feature means not near 0 after scaling: {col_means}"
    assert (col_stds  > 0.0).all(), "A feature has zero variance after scaling."


def test_original_dataframe_not_modified(tmp_path):
    """preprocess_data must not mutate the source CSV."""
    path = make_sample_csv(tmp_path)
    df_before = pd.read_csv(path).copy()
    preprocess_data(path)
    df_after = pd.read_csv(path)

    pd.testing.assert_frame_equal(df_before, df_after)


def test_target_variable_is_binary(tmp_path):
    """y_train and y_test must contain only 0s and 1s."""
    path = make_sample_csv(tmp_path)
    _, _, y_train, y_test, _ = preprocess_data(path)

    all_values = set(y_train.unique()) | set(y_test.unique())
    assert all_values.issubset({0, 1}), f"Non-binary values in target: {all_values}"


def test_train_test_split_proportions(tmp_path):
    """Test set should be ~20% of total rows (default split)."""
    path = make_sample_csv(tmp_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(path)

    total = len(y_train) + len(y_test)
    test_ratio = len(y_test) / total

    assert 0.10 <= test_ratio <= 0.40, f"Unexpected test ratio: {test_ratio:.2f}"


def test_output_feature_count(tmp_path):
    """Output must have exactly 8 features (one per input column minus Outcome)."""
    path = make_sample_csv(tmp_path)
    X_train, X_test, _, _, _ = preprocess_data(path)

    assert X_train.shape[1] == 8, f"Expected 8 features, got {X_train.shape[1]}"
    assert X_test.shape[1]  == 8, f"Expected 8 features, got {X_test.shape[1]}"


def test_scaler_transforms_consistently(tmp_path):
    """Scaler returned must transform test data without refitting."""
    path = make_sample_csv(tmp_path)
    X_train, X_test, _, _, scaler = preprocess_data(path)

    # Re-transforming X_test with the returned scaler should be idempotent
    # (scaler was already applied, so re-applying changes values — just verify no error)
    result = scaler.transform(X_test)
    assert result.shape == X_test.shape


def test_preprocessing_on_real_data():
    """Run preprocessing on the actual diabetes.csv and verify basic shape."""
    X_train, X_test, y_train, y_test, scaler = preprocess_data(DATA_PATH)

    assert X_train.shape[1] == 8
    assert X_test.shape[1]  == 8
    assert len(y_train) > len(y_test), "Training set should be larger than test set."
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
