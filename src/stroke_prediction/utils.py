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

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw stroke dataset from CSV.
    """
    return pd.read_csv(csv_path)


def make_xy(df: pd.DataFrame):
    """
    Preprocess the dataset and return features X and target y.

    This function must be deterministic and reusable.
    """
    df = df.copy()

    # Drop ID column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Target
    y = df["stroke"]
    X = df.drop(columns=["stroke"])

    # Column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Preprocessing pipelines
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
