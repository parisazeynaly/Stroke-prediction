# src/stroke_prediction/train.py

from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from stroke_prediction.utils import load_data, make_xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"


def main():
    print("[train] Loading data...")
    df = load_data(str(DATA_PATH))

    print("[train] Preprocessing (fit preprocessor)...")
    X_all, y, preprocessor = make_xy(df)  # fit happens here (in utils)

    print("[train] Splitting...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    X_train = X_all[train_idx]
    y_train = y.iloc[train_idx]

    print("[train] Training Logistic Regression (class_weight=balanced)...")
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)

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
