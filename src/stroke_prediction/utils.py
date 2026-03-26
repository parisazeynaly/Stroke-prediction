# src/stroke_prediction/utils.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw stroke dataset from CSV."""
    return pd.read_csv(csv_path)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build (but do NOT fit) the preprocessing pipeline for features X.
    Call .fit_transform(X_train) during training, then .transform(X_test) during eval.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def make_xy(df: pd.DataFrame):
    """
    Split df into features X and target y. Does NOT fit a preprocessor.
    Returns X (DataFrame), y (Series).
    """
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    y = df["stroke"]
    X = df.drop(columns=["stroke"])
    return X, y
