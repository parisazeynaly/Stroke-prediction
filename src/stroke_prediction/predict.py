
# src/stroke_prediction/predict.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_artifacts():
    """Load the saved model and preprocessor from outputs/."""
    model = joblib.load(OUTPUT_DIR / "model.joblib")
    preprocessor = joblib.load(OUTPUT_DIR / "preprocessor.joblib")
    return model, preprocessor


def predict_single(patient_dict: dict) -> dict:
    """
    Run inference on a single patient record.

    Parameters
    ----------
    patient_dict : dict
        Raw feature values, e.g.:
        {
            "gender": "Male", "age": 67, "hypertension": 0,
            "heart_disease": 0, "ever_married": "Yes",
            "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 228.69, "bmi": 36.6,
            "smoking_status": "formerly smoked"
        }

    Returns
    -------
    dict with keys: stroke_probability, no_stroke_probability, prediction (0 or 1)
    """
    model, preprocessor = load_artifacts()

    # Build a single-row DataFrame so the preprocessor works correctly
    df = pd.DataFrame([patient_dict])

    # Apply the saved preprocessor (fitted on training data — no leakage)
    X = preprocessor.transform(df)

    stroke_prob = float(model.predict_proba(X)[0, 1])
    no_stroke_prob = 1.0 - stroke_prob
    prediction = int(stroke_prob >= 0.5)

    return {
        "stroke_probability": round(stroke_prob, 4),
        "no_stroke_probability": round(no_stroke_prob, 4),
        "prediction": prediction,
    }
