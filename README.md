# Diabetes Risk Prediction

## Project Description
Build a trained ML classification Model that predicts whether a patient has diabetes based on clinical measurements. It is aimed at healthcare as a assistant in quick, explainable screening. 
Build a LLM powered interface (Nebius AI Studio)

Problem it solves: Diabetes often goes undetected until complications arise. The model that flags high-risk individuals from routine measurements (glucose, BMI, blood pressure, etc.) allows earlier detection and intervention.

## About Dataset (Classification Dataset)
Context

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
Content

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
Task : build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

Predict the onset of diabetes based on medical and demographic data such as glucose levels, BMI, and age

The project includes:
- A full ML pipeline training and comparing 6 model types
- MLflow experiment tracking
- A natural language chat interface powered by Nebius AI Studio (Qwen) where users can describe their health in plain English and receive a prediction in plain English
- Data drift detection using Evidently to monitor feature distribution shifts over time

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
this file is excluded by `.gitignore`.

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
Compares feature distributions between a reference dataset and new incoming data using the Kolmogorov-Smirnov test. Outputs an HTML report to `reports/drift_check_report.html`. See [drift_analysis.md](drift_analysis.md) for full findings.

---

## Architecture Overview

```
Nebius AI(Qwen) feature extraction 
    LLM will call for the features, and user will input the following features. LLM will ask follow up questions if there isnt enough data. 
        │
        ▼
StandardScaler
    This will apply the scaled features from training. Without this the model will see out of range values
        │
        ▼
The Best ML Model will make the prediction 
    Outputs the prediction with proability score
        │
        ▼
Nebius AI (generate)
    LLM will turn the raw numbers into clear plain english

```

**Training pipeline:** `preprocess.py` → `train.py` → `evaluation.py`  
All 6 models are trained, evaluated, and compared. The model with the best ROC-AUC score is saved alongside its scaler so inference is consistent with training.

**Drift monitoring:** `detect_drift.py` uses Evidently to run statistical tests between a reference dataset and new data. Results are thresholded into OK / Warning / Critical and saved as an HTML report.

---

## Results Summary

Performed a 80/20 train test split
KNN imputation to avoid bias instead of using median and standardScaler preprocessing

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

**Best model: XGBoost** — with ROC-AUC score of (0.8551).

**Key findings:**
- Very surpised that XGBoost was the best performing model, I was going to assume random_forest would perform the best. They are both very close.
- The accuracy is above .7 but is still rather concerning that its low for the medical field. 
- One concerning issue is low recall. False Negative is not good. Neural_network had the highest recall but was still falls under .7
- KNN imputation of zero values meaningfully improved recall compared to dropping missing rows or replace with median

---

## Reflection

**What I learned:**  
- Building an end to end ML pipeline.
- deciding which EDA to perform for this data set.
- Scaling the data and model alongside users input.
- Structuring configs so they are not hardcoded
- Using MLFlow hlped with tracking every run, comparing models, and reproducing results


- Integrating an LLM  was the most confusing part. The two-LLM-call pattern and pre prompting was very new to me. 
    ( extract -> predict -> explain ) is reusable for other classification problems.

**What was challenging:**  
- Getting the LLM to output valid JSON for all 8 features required careful prompt engineering 
- Structung configs was still very confusing and connecting all the files was confusing and overwhelming instead of just having one big file.
- Having to go back to sprints such as docker and streamlit.
- Building this with multiple configurations was confusing and hard to keep track
- I also faced a lot of file path issues with wrong names


**What I would improve with more time:**  
- Hyperparameter tuning to improve model
- Properly deploy docker, I feel like im missing some parts.
- Performce a quality check on Nebius AI accuracy and output when handling edge cases and to just have a number representation on confidence.
- File organization could be improved, Having drift in its own file? Should I combine drift analysis with README.md?
- When deploying There are some files that were created, I still do not understand what they do.