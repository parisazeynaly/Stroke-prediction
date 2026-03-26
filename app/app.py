# app/app.py

import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Make the src package importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction.predict import predict_single  # noqa: E402

app = Flask(__name__, template_folder="../templates")
logging.basicConfig(level=logging.INFO)

# Optional: Google Gemini for LLM explanations
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
llm_model = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel("gemini-pro")
        app.logger.info("Gemini API configured.")
    except Exception as e:
        app.logger.warning(f"Gemini init failed: {e}")


def risk_label(prob: float) -> str:
    if prob < 0.10:
        return "Very Low"
    elif prob < 0.30:
        return "Low"
    elif prob < 0.50:
        return "Moderate"
    elif prob < 0.70:
        return "High"
    return "Very High"


def build_llm_prompt(patient: dict, result: dict) -> str:
    p = patient
    return f"""
A machine learning model assessed a patient:
- Gender: {p.get('gender')}, Age: {p.get('age')}, Hypertension: {p.get('hypertension')},
  Heart disease: {p.get('heart_disease')}, Married: {p.get('ever_married')},
  Work: {p.get('work_type')}, Residence: {p.get('Residence_type')},
  Glucose: {p.get('avg_glucose_level')} mg/dL, BMI: {p.get('bmi')},
  Smoking: {p.get('smoking_status')}

Stroke probability: {result['stroke_probability']:.1%} ({risk_label(result['stroke_probability'])} risk)

Provide a concise, plain-language explanation of the key risk factors and general
lifestyle advice for stroke prevention. Do NOT give medical diagnoses or treatment advice.
Always recommend consulting a qualified healthcare professional.
""".strip()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        # Map incoming form keys to the feature names the preprocessor expects
        patient = {
            "gender":             data.get("gender", "Other"),
            "age":                float(data.get("age", 0)),
            "hypertension":       int(data.get("hypertension", 0)),
            "heart_disease":      int(data.get("disease", 0)),
            "ever_married":       data.get("married", "No"),
            "work_type":          data.get("work", "Private"),
            "Residence_type":     data.get("residence", "Urban"),
            "avg_glucose_level":  float(data.get("glucose", 0.0)),
            "bmi":                float(data.get("bmi", 0.0)),
            "smoking_status":     data.get("smoking", "Unknown"),
        }

        # Use the saved preprocessor — no manual one-hot encoding needed
        result = predict_single(patient)
        result["risk_level"] = risk_label(result["stroke_probability"])

        # Optional LLM explanation
        llm_explanation = None
        if llm_model:
            try:
                prompt = build_llm_prompt(patient, result)
                llm_explanation = llm_model.generate_content(prompt).text
            except Exception as e:
                app.logger.warning(f"LLM call failed: {e}")
                llm_explanation = "AI explanation unavailable."

        return jsonify({**result, "llm_explanation": llm_explanation})

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
