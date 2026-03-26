# src/stroke_prediction/train.py

from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from stroke_prediction.utils import load_data, make_xy, make_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"


def main():
    print("[train] Loading data...")
    df = load_data(str(DATA_PATH))

    print("[train] Splitting features and target...")
    X, y = make_xy(df)

    # Split BEFORE fitting the preprocessor to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test indices for reproducible evaluation
    test_idx = y_test.index.tolist()

    print("[train] Fitting preprocessor on training data only...")
    preprocessor = make_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)

    print("[train] Training Logistic Regression (class_weight=balanced)...")
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    model.fit(X_train_processed, y_train)

    print("[train] Saving artifacts to outputs/ ...")
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    joblib.dump(test_idx, OUTPUT_DIR / "test_idx.joblib")

    print("[train] Done. Saved:")
    print("  outputs/model.joblib")
    print("  outputs/preprocessor.joblib")
    print("  outputs/test_idx.joblib")


if __name__ == "__main__":
    main()
