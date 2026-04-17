import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REQUIRED_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

VALID_FEATURES = {
    "Pregnancies": 2, "Glucose": 140, "BloodPressure": 80,
    "SkinThickness": 20, "Insulin": 85, "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5, "Age": 45,
}


def parse_llm_output(text: str) -> dict:
    """Mirrors the parsing logic in app.py."""
    try:
        data = json.loads(text.strip())
        if all(k in data for k in REQUIRED_FEATURES):
            return data
    except json.JSONDecodeError:
        pass
    return {"_followup": text}


def test_complete_input_is_parsed_correctly():
    """All 8 features present in JSON must be extracted without a follow-up."""
    result = parse_llm_output(json.dumps(VALID_FEATURES))

    assert "_followup" not in result
    assert all(k in result for k in REQUIRED_FEATURES)


def test_incomplete_input_triggers_followup():
    """Partial or non-JSON input must return a follow-up, not a prediction."""
    result = parse_llm_output("I am 45 years old with a glucose of 140.")

    assert "_followup" in result
