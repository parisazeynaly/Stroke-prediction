#  Stroke Prediction with MLflow & Docker

This project predicts the likelihood of a stroke using patient data and machine learning. It includes full MLOps practices such as experiment tracking, model registry, and deployment with Docker.

## Dataset

- Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- Features: age, gender, hypertension, heart disease, BMI, smoking status, etc.

##  Features

✅ ML pipeline with Scikit-learn  
✅ Exploratory Data Analysis (EDA)  
✅ Model training and evaluation  
✅ MLflow experiment tracking  
✅ Model registry with MLflow  
✅ Adversarial testing (optional)  
✅ Dockerized Flask app for predictions  
✅ Web UI for user input

## Tech Stack

- Python 3.11
- Scikit-learn
- Pandas / Seaborn / Matplotlib
- MLflow (experiment tracking, model registry)
- Docker (deployment)
- Flask (Web API)

## How to Run

### Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction

Setup Environment
bash
Copy
Edit
pip install -r requirements.txt


Train Models and Log with MLflow:  python train.py
mlflow UI

Run Docker App:
docker build -t stroke-app .
docker run -p 5000:5000 stroke-app

Project Structure

├── app.py              # Flask app for user input and prediction
├── train.py            # Model training and MLflow logging
├── Dockerfile          # For containerization
├── templates/
│   └── index.html      # Frontend form
├── model_pickle.pkl    # Saved best model
├── scaler.pkl          # Saved StandardScaler
├── requirements.txt    # Python dependencies
└── README.md

Contact:
Feel free to reach out if you have any questions!

Future Work:
 Fairness testing using AIF360

⬜ Add model serving via MLflow REST API

⬜ Automate training with GitHub Actions

