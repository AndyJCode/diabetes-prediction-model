"""
Diabetes Risk Prediction — Streamlit chat app powered by Nebius AI Studio + a trained ML model.

How it works:
  1. User describes themselves in plain English.
  2. DeepSeek (via Nebius) extracts any new feature values from each message.
  3. Extracted values accumulate in session state across turns.
  4. Once all 8 features are collected, the best trained model runs inference.
  5. DeepSeek turns the raw prediction into a clear, human-friendly explanation.
"""

import os
import json
import numpy as np
import streamlit as st
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from train import load_best_model

# ── Constants ────────────────────────────────────────────────────────────────

NEBIUS_MODEL = "deepseek-ai/DeepSeek-V3.2"

REQUIRED_FEATURES = {
    "Pregnancies":              "number of pregnancies",
    "Glucose":                  "glucose level (mg/dL)",
    "BloodPressure":            "blood pressure (mm Hg)",
    "SkinThickness":            "skin thickness (mm)",
    "Insulin":                  "insulin level (μU/mL)",
    "BMI":                      "BMI (body mass index)",
    "DiabetesPedigreeFunction": "diabetes pedigree function (family history score)",
    "Age":                      "age (years)",
}

# ── LLM helpers ──────────────────────────────────────────────────────────────

def extract_new_values(client: OpenAI, user_message: str, known: dict) -> dict:
    """
    Ask the LLM to extract any feature values mentioned in the latest user message.
    Returns a dict of newly found feature name → value (may be empty if none found).
    Already-known features are listed so the model knows what's still needed.
    """
    feature_list = "\n".join(
        f"  - {name}: {desc}" for name, desc in REQUIRED_FEATURES.items()
    )
    known_str = json.dumps(known) if known else "none yet"
    missing = [name for name in REQUIRED_FEATURES if name not in known]
    missing_str = ", ".join(missing) if missing else "none — all collected"

    system_prompt = f"""You are a medical data extraction assistant.
Extract values for these 8 diabetes features from the user's message:
{feature_list}

Already collected: {known_str}
Still needed: {missing_str}

Rules:
- Extract ONLY values the user explicitly states in this message.
- NEVER guess or infer values not clearly stated.
- Pregnancies must be 0 if the user says they are male.
- Respond with ONLY a JSON object of newly found values. Example: {{"Glucose": 140, "Age": 45}}
- If nothing new is extractable, respond with {{}}
- Do not include already-collected features unless the user is correcting them."""

    response = client.chat.completions.create(
        model=NEBIUS_MODEL,
        max_tokens=256,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    text = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def ask_for_missing(client: OpenAI, known: dict, conversation: list) -> str:
    """Ask the LLM to generate a friendly follow-up question for the missing features."""
    missing = {
        name: desc for name, desc in REQUIRED_FEATURES.items() if name not in known
    }
    missing_str = "\n".join(f"  - {name}: {desc}" for name, desc in missing.items())
    known_str = "\n".join(f"  - {name}: {known[name]}" for name in known)

    system_prompt = f"""You are a friendly medical assistant collecting health information.

Already collected:
{known_str if known_str else '  (nothing yet)'}

Still needed:
{missing_str}

Ask the user for the missing values in a single, short, friendly message.
Do not repeat values already collected. Do not make up values."""

    response = client.chat.completions.create(
        model=NEBIUS_MODEL,
        max_tokens=256,
        temperature=0.6,
        messages=[{"role": "system", "content": system_prompt}]
        + conversation[-4:],  # last 2 exchanges for tone context
    )
    return (response.choices[0].message.content or "").strip()


def explain_prediction(client: OpenAI, features: dict, prediction: int,
                       probability: float) -> str:
    """Ask the LLM to explain the model's prediction in plain English."""
    risk = "high" if prediction == 1 else "low"
    prompt = f"""A diabetes risk model produced this result for a patient:
- Prediction: {risk} risk of diabetes
- Probability: {probability:.1%}
- Patient features: {json.dumps(features, indent=2)}

Write a clear, empathetic 3-4 sentence response that:
1. States the prediction result plainly.
2. Mentions 1-2 of the patient's specific values that most influenced the result.
3. Recommends they consult a healthcare professional regardless of the outcome.
4. Reminds them this is a screening tool, not a diagnosis."""

    response = client.chat.completions.create(
        model=NEBIUS_MODEL,
        max_tokens=512,
        temperature=0.6,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()

# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, scaler, features: dict) -> tuple[int, float]:
    """Scale raw feature values and run the ML model."""
    ordered = [features[k] for k in REQUIRED_FEATURES]
    X = np.array(ordered).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    return prediction, probability

# ── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")
    st.title("Diabetes Risk Prediction")
    st.caption("Describe your health information in plain English and I'll assess your diabetes risk.")

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        st.error("NEBIUS_API_KEY environment variable is not set.")
        st.stop()

    @st.cache_resource
    def get_model():
        return load_best_model("best_model.pkl")

    try:
        model, scaler = get_model()
    except FileNotFoundError:
        st.error("No trained model found. Run `python src/train.py` first.")
        st.stop()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "known_features" not in st.session_state:
        st.session_state.known_features = {}

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Opening message on first load
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I can assess your diabetes risk based on a few health measurements. "
                "Please describe yourself — for example: *'I'm 45 years old, BMI of 28, "
                "glucose 140, blood pressure 80, 2 pregnancies, skin thickness 20, "
                "insulin 85, pedigree 0.5'*. Share as much as you like and I'll ask "
                "for anything that's missing."
            )

    # Handle user input
    if user_input := st.chat_input("Describe your health information..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Extract any new values from this message and merge with what we know
                new_values = extract_new_values(
                    client, user_input, st.session_state.known_features
                )
                st.session_state.known_features.update(new_values)
                known = st.session_state.known_features

            if len(known) == len(REQUIRED_FEATURES):
                # All 8 features collected — run the ML model
                prediction, probability = run_inference(model, scaler, known)

                with st.spinner("Generating explanation..."):
                    explanation = explain_prediction(client, known, prediction, probability)

                if prediction == 1:
                    st.error(f"**High diabetes risk** ({probability:.1%} probability)")
                else:
                    st.success(f"**Low diabetes risk** ({probability:.1%} probability)")

                st.markdown(explanation)

                with st.expander("Features used for prediction"):
                    for name, desc in REQUIRED_FEATURES.items():
                        st.write(f"**{desc}:** {known[name]}")

                st.session_state.messages.append({"role": "assistant", "content": explanation})

            else:
                # Still missing values — ask for them
                collected = len(known)
                total = len(REQUIRED_FEATURES)

                with st.spinner("..."):
                    reply = ask_for_missing(client, known, st.session_state.messages)

                progress = f"*({collected}/{total} values collected)*\n\n"
                st.markdown(progress + reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": progress + reply}
                )

    if st.session_state.messages and st.button("Start over"):
        st.session_state.messages = []
        st.session_state.known_features = {}
        st.rerun()


if __name__ == "__main__":
    main()
