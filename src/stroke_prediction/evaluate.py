# src/stroke_prediction/evaluate.py

from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from stroke_prediction.utils import load_data, make_xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"


def main():
    print("[eval] Loading data...")
    df = load_data(str(DATA_PATH))

   
    X, y, _ = make_xy(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[eval] Loading model...")
    model = joblib.load(OUTPUT_DIR / "model.joblib")

    print("[eval] Predicting...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))

    out_path = REPORTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[eval] Wrote {out_path}")
    print("[eval] Metrics:", metrics)


if __name__ == "__main__":
    main()

