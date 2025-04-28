from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

model = pickle.load(open("model_pickle.pkl", 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        disease = int(request.form['disease'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']

        # gender
        if gender == "Male":
            gender_male = 1
            gender_other = 0
            gender_female = 0
        elif gender == "Other":
            gender_male = 0
            gender_other = 1
            gender_female = 0
        else:  # Female
            gender_male = 0
            gender_other = 0
            gender_female = 1

        # married
        married_yes = 1 if married == "Yes" else 0

        # work type
        work_type_Never_worked = 1 if work == "Never_worked" else 0
        work_type_Private = 1 if work == "Private" else 0
        work_type_Self_employed = 1 if work == "Self-employed" else 0
        work_type_children = 1 if work == "children" else 0

        # residence type
        Residence_type_Urban = 1 if residence == "Urban" else 0

        # smoking status
        smoking_status_formerly_smoked = 1 if smoking == 'formerly smoked' else 0
        smoking_status_never_smoked = 1 if smoking == 'never smoked' else 0
        smoking_status_smokes = 1 if smoking == 'smokes' else 0

        feature = scaler.fit_transform([[age, hypertension, disease, glucose, bmi,
                                         gender_male, gender_other, gender_female,
                                         married_yes,
                                         work_type_Never_worked, work_type_Private,
                                         work_type_Self_employed, work_type_children,
                                         Residence_type_Urban,
                                         smoking_status_formerly_smoked, smoking_status_never_smoked,
                                         smoking_status_smokes]])

        prediction = model.predict(feature)[0]

        prediction = "YES" if prediction == 1 else "NO"

        return render_template("index.html", prediction_text=f"Chance of Stroke Prediction is --> {prediction}")

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
