# Stroke Risk Prediction: Project Overview
A machine learning web application that predicts the likelihood of a person having a stroke based on health-related inputs. Built using Flask, Scikit-learn, Docker, and deployed with CI/CD via GitHub Actions and Render.
Deployment Link: https://stroke-prediction-2brs.onrender.com

Code and Resources Used:
Python Version: 3.7
Packages: pandas, numpy, sklearn, matplotlib, seaborn, imblearn, flask, pickle, plotly.

Business Problem / Objective:
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Use this dataset to predict whether a patient is likely to get a stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.


For Web Framework Requirements: pip install -r requirements.txt
Predict stroke risk based on user inputs (age, hypertension, heart disease, etc.)

Built with RandomForestClassifier and serialized using pickle

Web frontend using Flask & HTML templates

Containerized using Docker

Automated CI/CD pipeline with GitHub Actions

Deployed on Render.

⚙️ Setup Instructions

git clone https://github.com/parisazeynaly/Stroke-prediction.git

Install dependencies
pip install -r requirements.txt

Run the app
python app.py
Then open http://localhost:5000

Docker (optional)

Build: docker build -t stroke-predictor .

Run: docker run -p 5000:5000 stroke-predictor

🧪 ML Model
Algorithm: RandomForestClassifier

Preprocessing: Standard scaling, feature selection

Evaluation: Accuracy, confusion matrix

Saved as: model_pickle.pkl

🔄 CI/CD Workflow
On every push, GitHub Actions:

Runs Docker build

Checks dependencies

Then auto-deploys via Render

Feel free to open an issue or contribute!



