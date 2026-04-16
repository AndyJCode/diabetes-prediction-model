"""
Diabetes Risk Prediction — Streamlit chat app powered by Nebius AI Studio + a trained ML model.

How it works:
  1. User describes themselves in plain English.
  2. Llama (via Nebius) extracts the 8 diabetes features as JSON (or asks for missing ones).
  3. The best trained model runs inference on the scaled features.
  4. Llama turns the raw prediction into a clear, human-friendly explanation.
"""

import os
import json
import numpy as np
import streamlit as st
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env file into environment variables

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from train import load_best_model

# ── Constants ────────────────────────────────────────────────────────────────

NEBIUS_MODEL = "Qwen/Qwen3.5-397B-A17B"

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

def extract_features(client: OpenAI, conversation: list) -> dict:
    """
    Ask Llama to extract diabetes features from the conversation so far.
    Returns a dict of feature name → value if all 8 are found,
    or {"_followup": "..."} if the model needs more information.

    Nebius uses the OpenAI-compatible format: the system prompt goes as the
    first message with role "system", followed by the conversation history.
    """
    feature_list = "\n".join(
        f"  - {name}: {desc}" for name, desc in REQUIRED_FEATURES.items()
    )
    system_prompt = f"""You are a medical data extraction assistant.
Your job is to extract exactly these 8 features from the user's messages:
{feature_list}

Rules:
- If you can extract ALL 8 features confidently, respond with ONLY valid JSON,
  no extra text. Example: {{"Pregnancies": 2, "Glucose": 140, ...}}
- If any feature is missing or unclear, respond with a friendly question asking
  only for the missing information. Do NOT produce JSON until all 8 are known.
- Never make up or guess values. Only use what the user explicitly stated.
- Pregnancies must be 0 for males — if the user says they are male, set it to 0."""

    messages = [{"role": "system", "content": system_prompt}] + conversation

    response = client.chat.completions.create(
        model=NEBIUS_MODEL,
        max_tokens=512,
        temperature=0.6,
        top_p=0.65,
        messages=messages,
    )
    text = response.choices[0].message.content.strip()

    try:
        data = json.loads(text)
        if all(k in data for k in REQUIRED_FEATURES):
            return data
    except json.JSONDecodeError:
        pass

    return {"_followup": text}


def explain_prediction(client: OpenAI, features: dict, prediction: int,
                       probability: float) -> str:
    """
    Ask Llama to explain the model's prediction in plain English.
    This is a single-turn call — no conversation history needed.
    """
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
        max_tokens=300,
        temperature=0.6,
        top_p=0.65,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, scaler, features: dict) -> tuple[int, float]:
    """
    Scale raw feature values using the training scaler, then predict.
    The scaler must match the one used during training — that's why we save
    and load it alongside the model in best_model.pkl.
    """
    ordered = [features[k] for k in REQUIRED_FEATURES]
    X = np.array(ordered).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    return prediction, probability

# ── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")
    st.title("🩺 Diabetes Risk Prediction")
    st.caption("Describe your health information in plain English and I'll assess your diabetes risk.")

    # Check for API key
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        st.error("NEBIUS_API_KEY environment variable is not set.")
        st.stop()

    # Load model once and cache it across reruns
    @st.cache_resource
    def get_model():
        return load_best_model("best_model.pkl")

    try:
        model, scaler = get_model()
    except FileNotFoundError:
        st.error("No trained model found. Run `python src/train.py` first to train and save a model.")
        st.stop()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    )

    # Session state holds the full conversation for multi-turn context
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                result = extract_features(client, st.session_state.messages)

            if "_followup" in result:
                # LLM needs more information — show its follow-up question
                reply = result["_followup"]
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            else:
                # All 8 features extracted — run the ML model
                prediction, probability = run_inference(model, scaler, result)

                with st.spinner("Generating explanation..."):
                    explanation = explain_prediction(client, result, prediction, probability)

                if prediction == 1:
                    st.error(f"**High diabetes risk** ({probability:.1%} probability)")
                else:
                    st.success(f"**Low diabetes risk** ({probability:.1%} probability)")

                st.markdown(explanation)

                with st.expander("Features used for prediction"):
                    for name, desc in REQUIRED_FEATURES.items():
                        st.write(f"**{desc}:** {result[name]}")

                st.session_state.messages.append({"role": "assistant", "content": explanation})

    if st.session_state.messages and st.button("Start over"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
