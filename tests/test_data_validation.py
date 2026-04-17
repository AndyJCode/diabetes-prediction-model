import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

DATA_PATH = str(Path(__file__).parent.parent / 'data' / 'diabetes.csv')

REQUIRED_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def test_required_columns():
    """Dataset must contain all 9 expected columns."""
    df = load_dataset()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_target_variable_is_binary():
    """Outcome column must only contain 0 and 1."""
    df = load_dataset()
    unique = set(df['Outcome'].unique())
    assert unique.issubset({0, 1}), f"Unexpected values in Outcome: {unique}"


def test_no_fully_null_rows():
    """No row should be entirely null."""
    df = load_dataset()
    fully_null = df.isnull().all(axis=1).sum()
    assert fully_null == 0, f"{fully_null} fully-null rows found."


def test_dataset_has_minimum_rows():
    """Dataset must have at least 500 rows to be usable for training."""
    df = load_dataset()
    assert len(df) >= 500, f"Dataset too small: {len(df)} rows."


def test_feature_columns_are_numeric():
    """All feature columns must be numeric."""
    df = load_dataset()
    for col in REQUIRED_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric."
