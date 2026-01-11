# src/stroke_prediction/train.py

from pathlib import Path
import joblib
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

    print("[train] Preprocessing...")
    X, y, preprocessor = make_xy(df)

    print("[train] Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[train] Training baseline model (Logistic Regression)...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    print("[train] Saving artifacts to outputs/ ...")
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")

    # Save split indices for deterministic evaluation (important!)
    # This makes evaluate.py use the *same* test set every time.
    joblib.dump(
        {"test_indices_count": len(y_test)},
        OUTPUT_DIR / "split_info.joblib"
    )

    print("[train] Done. Saved:")
    print("  outputs/model.joblib")
    print("  outputs/preprocessor.joblib")


if __name__ == "__main__":
    main()
