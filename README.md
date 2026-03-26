# Stroke Prediction — ML Pipeline

A reproducible machine learning pipeline to predict stroke risk from demographic and clinical data.
Demonstrates end-to-end ML engineering: modular src layout, no data leakage, MLflow tracking, Docker deployment, and a Flask web app with optional LLM explanations.

---

## Dataset

- **Source:** [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Size:** ~5,000 records, 11 features (age, BMI, hypertension, smoking status, etc.)
- Download the CSV and place it at `data/healthcare-dataset-stroke-data.csv`

---

## Quick start

```bash
git clone https://github.com/parisazeynaly/Stroke-prediction.git
cd Stroke-prediction
pip install -r requirements.txt

make train      # fit preprocessor + model, save to outputs/
make eval       # evaluate on held-out test set, save reports/
make run-api    # start Flask app at http://localhost:5000
```

Or run the full pipeline in one step:

```bash
make reproduce
```

---

## Project structure

```
Stroke-prediction/
├── src/stroke_prediction/
│   ├── utils.py        # load_data, make_xy, make_preprocessor
│   ├── train.py        # fit + save model and preprocessor
│   ├── evaluate.py     # threshold tuning, metrics, calibration curve
│   └── predict.py      # single-record inference using saved artifacts
├── app/
│   └── app.py          # Flask API
├── templates/
│   └── index.html      # web UI
├── data/               # put the CSV here (gitignored)
├── outputs/            # saved model + preprocessor (gitignored)
├── reports/            # metrics.json, classification_report.txt, plots
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Methods

**Preprocessing** (fit on training data only — no leakage):
- KNN imputation for missing BMI values
- Standard scaling for numeric features
- One-hot encoding for categorical features

**Model:** Logistic Regression with `class_weight="balanced"` to handle severe class imbalance (~5% positive rate).

**Evaluation:**
- ROC-AUC and PR-AUC (appropriate for imbalanced data)
- Threshold tuning via precision-recall curve to maximise F1
- Brier score and calibration curve

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.843 |
| PR-AUC | 0.268 |
| F1 @best threshold | 0.359 |
| Best threshold | 0.844 |
| Recall @best threshold | 0.420 |

> Note: accuracy is not a meaningful metric here due to the ~5% stroke rate.
> PR-AUC and recall are the primary evaluation targets.

---

## Deployment

**Docker:**
```bash
docker build -t stroke-prediction .
docker run -p 5000:5000 stroke-prediction
```

**With LLM explanations (optional):**
```bash
docker run -p 5000:5000 -e GOOGLE_API_KEY=your_key stroke-prediction
```

---

## Design notes

- The `preprocessor.joblib` saved at training time is the only object used at inference time — no manual feature engineering in the app.
- Test indices are saved alongside the model so evaluation is always on the exact same held-out split.
- `make reproduce` runs the full train → eval pipeline deterministically (random seed fixed).
