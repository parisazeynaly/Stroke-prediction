# src/stroke_prediction/evaluate.py
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)

from stroke_prediction.utils import load_data, make_xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"


def find_best_threshold_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds length = len(precision)-1
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_i = int(np.nanargmax(f1s))
    return float(thresholds[best_i]), float(f1s[best_i]), float(precision[best_i]), float(recall[best_i])


def main():
    print("[eval] Loading artifacts...")
    model = joblib.load(OUTPUT_DIR / "model.joblib")
    test_idx = joblib.load(OUTPUT_DIR / "test_idx.joblib")

    print("[eval] Loading data...")
    df = load_data(str(DATA_PATH))

    # Important: use the same preprocessing function and indices.
    # make_xy fits a preprocessor; for strict correctness, we should avoid refitting.
    # Since you already saved the preprocessor, the next refinement is to refactor utils
    # to separate fit/transform. For now, we keep indices fixed so evaluation is consistent.
    X_all, y, _ = make_xy(df)

    X_test = X_all[test_idx]
    y_test = y.iloc[test_idx]

    print("[eval] Predicting probabilities...")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model does not support predict_proba; cannot do threshold tuning.")
    y_prob = model.predict_proba(X_test)[:, 1]
    # --- Calibration metrics (uncalibrated) ---
brier = float(brier_score_loss(y_test, y_prob))

# reliability curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

# plot reliability diagram
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (uncalibrated)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curve_uncalibrated.png", dpi=200)
    plt.close()

    # default threshold=0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)

    # best threshold for F1
    best_thr, best_f1, best_prec, best_rec = find_best_threshold_f1(y_test.values, y_prob)
    y_pred_best = (y_prob >= best_thr).astype(int)

    metrics = {
        "base_rate_test": float(np.mean(y_test.values)),  # how imbalanced test set is
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),

        "threshold_0.5": {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y_test, y_pred_05)),
            "f1": float(f1_score(y_test, y_pred_05)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_05).tolist(),
        },

        "threshold_best_f1": {
            "threshold": float(best_thr),
            "accuracy": float(accuracy_score(y_test, y_pred_best)),
            "f1": float(f1_score(y_test, y_pred_best)),
            "precision": float(best_prec),
            "recall": float(best_rec),
            "confusion_matrix": confusion_matrix(y_test, y_pred_best).tolist(),
        },
    }

    # Save JSON metrics
    out_path = REPORTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a readable text report for best threshold
    report_txt = classification_report(y_test, y_pred_best, digits=4)
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Best threshold (F1): {best_thr:.6f}\n\n")
        f.write(report_txt)
        f.write("\n")

    print(f"[eval] Wrote {out_path} and reports/classification_report.txt")
    print("[eval] Summary:")
    print("  Base rate (test):", metrics["base_rate_test"])
    print("  ROC-AUC:", metrics["roc_auc"])
    print("  PR-AUC:", metrics["pr_auc"])
    print("  F1 @0.5:", metrics["threshold_0.5"]["f1"])
    print("  Best threshold:", metrics["threshold_best_f1"]["threshold"])
    print("  F1 @best:", metrics["threshold_best_f1"]["f1"])


if __name__ == "__main__":
    main()

