# Diabetes Risk Prediction

## Project Description

This project builds a machine learning system that predicts whether a patient has diabetes based on clinical measurements. It is aimed at healthcare support scenarios where a quick, explainable screening tool is needed — not as a replacement for clinical diagnosis, but as an early-warning aid.

**Problem it solves:** Diabetes often goes undetected until complications arise. A model that flags high-risk individuals from routine measurements (glucose, BMI, blood pressure, etc.) allows earlier referral and intervention.

The project includes:
- A full ML pipeline training and comparing 6 model types
- MLflow experiment tracking
- A natural language chat interface powered by Nebius AI Studio (Qwen) where users can describe their health in plain English and receive a plain-English risk assessment

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the data
Download the [Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it at:
```
data/diabetes.csv
```

### 3. Configure your API key
Create a `.env` file in the project root:
```
NEBIUS_API_KEY=your-key-here
```
Never commit this file — it is excluded by `.gitignore`.

### 4. (Optional) Review model settings
All hyperparameters and model choices live in `configs/config.yaml`. Edit that file to add or tune models — no code changes needed.

---

## Usage

### Train all models and save the best one
```bash
python src/train.py
```
Trains all 6 models, logs results to MLflow, prints a comparison table, and saves the best model to `best_model.pkl`.

### Launch the chat app
```bash
python -m streamlit run app.py
```
Type your health information in plain English, e.g.:
> *"I'm 45 years old, BMI 28, glucose 140, blood pressure 80, 2 pregnancies, skin thickness 20, insulin 85, pedigree score 0.5"*

The app will ask for any missing information, run the model, and explain the result.

### View MLflow experiment dashboard
```bash
python -m mlflow ui
```
Open `http://localhost:5000` to compare all runs side by side.

### Run tests
```bash
pytest tests/
```

### Compare experiments from the command line
```bash
python compare_experiments.py
```

---

## Architecture Overview

```
User types natural language
        │
        ▼
┌─────────────────────────┐
│  Nebius AI (Qwen)       │  LLM Call 1 — parse natural language
│  Feature Extraction     │  into structured JSON with all 8 features.
│                         │  If any are missing, ask a follow-up question.
└────────────┬────────────┘
             │  {Pregnancies, Glucose, BMI, ...}
             ▼
┌─────────────────────────┐
│  StandardScaler         │  Apply the same scaler used during training.
│  (loaded from .pkl)     │  Without this, the model sees out-of-range values.
└────────────┬────────────┘
             │  scaled features
             ▼
┌─────────────────────────┐
│  Best ML Model          │  Outputs: prediction (0/1) + probability score.
│  (loaded from .pkl)     │
└────────────┬────────────┘
             │  prediction + probability
             ▼
┌─────────────────────────┐
│  Nebius AI (Qwen)       │  LLM Call 2 — turn the raw numbers into a
│  Explanation Generator  │  clear, empathetic explanation with caveats.
└────────────┬────────────┘
             │
             ▼
      Streamlit Chat UI
```

**Training pipeline:** `preprocess.py` → `train.py` → `evaluation.py`  
All 6 models are trained, evaluated, and compared. The best by ROC-AUC is saved alongside its scaler so inference is consistent with training.

---

## Results Summary

All models were evaluated on an 80/20 train/test split with KNN imputation and StandardScaler preprocessing.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.779 | 0.714 | 0.662 | 0.687 | 0.843 |
| Gradient Boosting | 0.766 | 0.698 | 0.649 | 0.673 | 0.831 |
| Logistic Regression | 0.753 | 0.681 | 0.635 | 0.657 | 0.819 |
| Neural Network | 0.760 | 0.692 | 0.622 | 0.655 | 0.824 |
| XGBoost | **0.792** | **0.731** | **0.676** | **0.702** | **0.857** |
| LightGBM | 0.779 | 0.714 | 0.649 | 0.680 | 0.841 |

**Best model: XGBoost** — selected by ROC-AUC (0.857).

**Key findings:**
- Tree-based ensemble methods (XGBoost, Random Forest, LightGBM) consistently outperformed the linear baseline
- Glucose level and BMI were the most influential features across models
- KNN imputation of zero values meaningfully improved recall compared to dropping missing rows

---

## Reflection

**What I learned:**  
Building an end-to-end ML pipeline — from raw CSV to a deployed chat interface — revealed how much complexity lives between training a model and making it useful. Saving the scaler alongside the model, structuring configs so hyperparameters are never hardcoded, and using MLflow to track every run all felt like overhead at first but made comparing experiments and reproducing results much easier.

Integrating an LLM for natural language parsing was the most interesting part. The two-LLM-call pattern (extract → predict → explain) is clean and reusable for other classification problems.

**What was challenging:**  
- Getting the LLM to reliably output valid JSON for all 8 features required careful prompt engineering
- The scaler/model pairing is easy to get wrong — raw feature values passed to a StandardScaler-trained model produce silently wrong predictions
- Multi-turn conversation state in Streamlit needed careful handling with `st.session_state`

**What I would improve with more time:**  
- Hyperparameter tuning with cross-validation (GridSearchCV or Optuna) rather than fixed defaults
- SHAP values to show which specific features drove each individual prediction
- A proper train/validation/test split to avoid any leakage from the imputer
- Dockerize the app for one-command deployment
