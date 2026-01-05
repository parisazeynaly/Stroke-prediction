import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, average_precision_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib,
import os

# Load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print("--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Dataset Information ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())


print("\n--- Missing Values Count per Column ---")
print(df.isna().sum())

print("\n--- Column Names ---")
print(df.columns.tolist())


### 3. Handle Missing Values (BMI)

# The 'bmi' column has missing values, which will be imputed using KNNImputer.
imputer = KNNImputer(n_neighbors=5)
df['bmi'] = imputer.fit_transform(df[['bmi']])

print("\n--- Missing Values after BMI Imputation ---")
print(df.isna().sum())
