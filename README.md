# 🫀 Stroke Prediction with Machine Learning

A reproducible ML pipeline to predict stroke risk using demographic and health data.  
This project demonstrates **end-to-end ML engineering** with **MLflow tracking**, **Dockerized deployment**, and a **Flask web application**.XAI

---

## 📌 Motivation
Stroke is one of the leading causes of death and disability worldwide.  
Early prediction using accessible health indicators can help preventive measures.  
This repository provides a **case study** on applying machine learning to structured health data, with a focus on reproducibility and deployment.

---

## 🗂️ Dataset
- **Source:** [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- **Size:** ~5,000 records, 11 features (e.g., age, BMI, hypertension, smoking status).  
- **Preprocessing:**
  - Missing value imputation for BMI
  - One-hot encoding for categorical features
  - Scaling numerical features

---

## ⚙️ Methods
We trained and compared several models:
- Logistic Regression
- Random Forest
- XGBoost

### Evaluation metrics:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC

---

## 📊 Results

| Model              | Accuracy | F1-score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression| 0.82     | 0.71     | 0.85    |
| Random Forest      | 0.86     | 0.75     | 0.90    |
| XGBoost            | 0.88     | 0.78     | 0.92    |

✅ **XGBoost achieved the best balance between recall and ROC-AUC**

---

## 🚀 Deployment

- **MLflow Tracking:** All experiments logged with parameters, metrics, and artifacts.  
- **Dockerized App:** Flask web interface for inputting patient data and predicting stroke risk.  
- **Demo Screenshot:**  
  ![App Demo](docs/demo.png)

---

## 🏗️ Project Structure
Stroke-prediction/
├── src/
│ ├── data_preprocessing.py
│ ├── stroke_prediction.py # main training & evaluation
│ └── utils.py
├── app/
│ ├── app.py # Flask web server
│ └── templates/
│ └── index.html
├── notebooks/ # EDA & model experiments
├── requirements.txt
├── Dockerfile
├── mlruns/ # MLflow artifacts
└── README.md
