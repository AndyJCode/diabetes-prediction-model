# Diabetes Risk Prediction

## What This Project Does

This project builds a machine learning model that predicts whether a patient has diabetes based on clinical measurements. The goal is to help with quick screening in healthcare settings where catching diabetes early can make a real difference.

On top of the ML model, I built a chat interface powered by Nebius AI Studio where users can just describe themselves in plain English. The LLM pulls out the clinical values from what they say, fills in gaps by asking follow-up questions, and then explains the prediction in plain English too.

**The problem it solves:** Diabetes often goes undetected until complications show up. A model that flags high-risk individuals from routine measurements like glucose, BMI, and blood pressure gives doctors an earlier heads up.

The project includes:
- A full ML pipeline that trains and compares 6 different model types
- MLflow experiment tracking so every run is logged and comparable
- A natural language chat interface where users describe their health info and the LLM handles the rest, including keeping track of what was already said across multiple messages
- Data drift detection using Evidently to monitor if feature distributions shift over time

---

## Dataset

This dataset comes from the National Institute of Diabetes and Digestive and Kidney Diseases. All patients are females at least 21 years old of Pima Indian heritage.

The predictor variables are: number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The target variable is `Outcome` (1 = diabetes, 0 = no diabetes).

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

### 3. Add your API key
Create a `.env` file in the project root:
```
NEBIUS_API_KEY=your-key-here
```
This file is excluded by `.gitignore` so it won't get pushed to GitHub.

### 4. (Optional) Adjust model settings
All hyperparameters and model choices are in `configs/config.yaml`. You can add or tune models there without touching any code.

---

## How to Run

### Train all models
```bash
python src/train.py
```
Trains all 6 models, logs everything to MLflow, prints a comparison table, and saves the best model to `best_model.pkl`.

### Launch the chat app
```bash
python -m streamlit run app.py
```
Type your health info in plain English, for example:
> *"I'm 45 years old, BMI 28, glucose 140, blood pressure 80, 2 pregnancies, skin thickness 20, insulin 85, pedigree score 0.5"*

The app asks for anything that's missing, runs the model, and explains the result.

### View the MLflow dashboard
```bash
python -m mlflow ui
```
Open `http://localhost:5000` to compare all 6 runs side by side.

### Run tests
```bash
python -m pytest tests/ -v
```

### Compare experiments from the command line
```bash
python compare_experiments.py
```

### Run drift detection
```bash
python detect_drift.py data/reference.csv data/current.csv
```
Compares feature distributions between a reference dataset and new incoming data using the Kolmogorov-Smirnov test. Outputs an HTML report to `reports/drift_check_report.html`. See [drift_analysis.md](drift_analysis.md) for the full findings.

---

## How It Works

```
User describes themselves in plain English
        │
        ▼
Nebius AI (DeepSeek) — feature extraction
    Pulls out clinical values from what the user says.
    Remembers what was already collected and asks only for what's missing.
        │
        ▼
StandardScaler
    Scales the values the same way the training data was scaled.
    Without this, the model would see out-of-range values.
        │
        ▼
Best ML Model (XGBoost)
    Makes the prediction and outputs a probability score.
        │
        ▼
Nebius AI (DeepSeek) — explanation
    Turns the raw numbers into a clear, plain English response.
```

**Training pipeline:** `preprocess.py` → `train.py` → `evaluation.py`

All 6 models are trained, evaluated, and compared. The model with the best ROC-AUC is saved alongside its scaler so inference stays consistent with how training was done.

**Drift monitoring:** `detect_drift.py` uses Evidently to run statistical tests between a reference dataset and new data. Results are bucketed into OK / Warning / Critical and saved as an HTML report.

---

## Results

80/20 train/test split. Used KNN imputation on zero values instead of median replacement or dropping rows, which meaningfully improved recall.

```
==================================================================================
Model                     accuracy   precision      recall    f1_score     roc_auc
==================================================================================
random_forest               0.7727      0.6667      0.5417      0.5977      0.8550
gradient_boosting           0.7922      0.7222      0.5417      0.6190      0.8439
logistic_regression         0.7597      0.7037      0.3958      0.5067      0.8333
neural_network              0.7857      0.6471      0.6875      0.6667      0.8235
xgboost                     0.7792      0.6842      0.5417      0.6047      0.8551
lightgbm                    0.7532      0.6136      0.5625      0.5870      0.8241
==================================================================================
```

**Best model: XGBoost** — ROC-AUC of 0.8551.

**Key findings:**
- I was honestly surprised XGBoost came out on top. I expected random forest to win and they ended up being really close. Neural network had the best F1 score.
- Accuracy above 0.7 sounds decent but for the medical field that's still a bit concerning. A wrong prediction here has real consequences.
- Recall is the one that worries me most. A false negative means someone who has diabetes gets told they probably don't. Neural network had the highest recall but still didn't break 0.7.
- KNN imputation made a noticeable difference compared to dropping rows or replacing zeros with the median.

---

## Reflection

**What I learned:**
- How to build a full end-to-end ML pipeline from preprocessing to deployment.
- How to think through which EDA steps actually matter for a given dataset.
- Why the scaler has to be saved alongside the model — otherwise inference breaks.
- How to structure configs so nothing is hardcoded and models can be swapped without touching the code.
- MLflow made it so much easier to track every run, compare models, and reproduce results.
- Integrating an LLM was the most confusing part for me. The two-call pattern (extract → predict → explain) was completely new. Once I got it working though, I realized it's actually a pretty reusable pattern for any classification problem.

**What was challenging:**
- Getting the LLM to output clean JSON for all 8 features took a lot of prompt engineering. The model kept either guessing missing values or wrapping the JSON in code fences.
- Keeping context across multiple chat turns was tricky. The LLM would lose values mentioned earlier in the conversation. I ended up storing extracted features in session state and only asking the LLM to pull values from the latest message.
- Figuring out which Nebius model to use. Some models are "thinking" models that return reasoning tokens instead of a normal response, which broke the app in a confusing way.
- Structuring the project across multiple files and configs was overwhelming at times. I kept running into path issues and mismatched names.
- Going back to Docker and Streamlit after not using them for a while.

**What I would improve with more time:**
- Hyperparameter tuning to push the model metrics higher, especially recall.
- Properly finishing the Docker deployment. I got it running but I feel like I'm missing some pieces.
- Running a quality check on the LLM responses for edge cases — things like unusual values or vague inputs.
- Cleaning up the file structure a bit. I'm still not sure if the drift analysis should live in its own file or be folded into the README.
