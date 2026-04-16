
'''About Dataset (Classification Dataset)
Context

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
Content

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
Task : build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

Predict the onset of diabetes based on medical and demographic data such as glucose levels, BMI, and age

'''


# Data cleaning and feature engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer



def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Identify columns with zero values that should be treated as missing
    cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace zero values with NaN
    df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

    # Impute missing values using KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Separate features and target variable
    X = df_imputed.drop('Outcome', axis=1)
    y = df_imputed['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    file_path = 'diabetes.csv'  # Update this path to your dataset
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    print("Preprocessing completed. Data is ready for model training.")


