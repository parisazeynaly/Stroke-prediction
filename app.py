# from flask import Flask, request, render_template
# import pickle
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
#
# model = pickle.load(open("model_pickle.pkl", 'rb'))
#
# app = Flask(__name__)
#
#
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == "POST":
#         gender = request.form['gender']
#         age = int(request.form['age'])
#         hypertension = int(request.form['hypertension'])
#         disease = int(request.form['disease'])
#         married = request.form['married']
#         work = request.form['work']
#         residence = request.form['residence']
#         glucose = float(request.form['glucose'])
#         bmi = float(request.form['bmi'])
#         smoking = request.form['smoking']
#
#         # gender
#         if gender == "Male":
#             gender_male = 1
#             gender_other = 0
#             gender_female = 0
#         elif gender == "Other":
#             gender_male = 0
#             gender_other = 1
#             gender_female = 0
#         else:  # Female
#             gender_male = 0
#             gender_other = 0
#             gender_female = 1
#
#         # married
#         married_yes = 1 if married == "Yes" else 0
#
#         # work type
#         work_type_Never_worked = 1 if work == "Never_worked" else 0
#         work_type_Private = 1 if work == "Private" else 0
#         work_type_Self_employed = 1 if work == "Self-employed" else 0
#         work_type_children = 1 if work == "children" else 0
#
#         # residence type
#         Residence_type_Urban = 1 if residence == "Urban" else 0
#
#         # smoking status
#         smoking_status_formerly_smoked = 1 if smoking == 'formerly smoked' else 0
#         smoking_status_never_smoked = 1 if smoking == 'never smoked' else 0
#         smoking_status_smokes = 1 if smoking == 'smokes' else 0
#
#         feature = scaler.fit_transform([[age, hypertension, disease, glucose, bmi,
#                                          gender_male, gender_other, gender_female,
#                                          married_yes,
#                                          work_type_Never_worked, work_type_Private,
#                                          work_type_Self_employed, work_type_children,
#                                          Residence_type_Urban,
#                                          smoking_status_formerly_smoked, smoking_status_never_smoked,
#                                          smoking_status_smokes]])
#
#         prediction = model.predict(feature)[0]
#
#         prediction = "YES" if prediction == 1 else "NO"
#
#         return render_template("index.html", prediction_text=f"Chance of Stroke Prediction is --> {prediction}")
#
#     else:
#         return render_template("index.html")
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, request, render_template
# import pickle
# import numpy as np
#
# app = Flask(__name__)
#
# model = pickle.load(open("model_pickle.pkl", 'rb'))
# scaler = pickle.load(open("scaler.pkl", "rb"))  # باید فایل scaler.pkl رو از مرحله آموزش ذخیره کرده باشی
#
# def preprocess_input(form):
#     gender = form['gender']
#     age = int(form['age'])
#     hypertension = int(form['hypertension'])
#     disease = int(form['disease'])
#     married = form['married']
#     work = form['work']
#     residence = form['residence']
#     glucose = float(form['glucose'])
#     bmi = float(form['bmi'])
#     smoking = form['smoking']
#
#     gender_male = 1 if gender == "Male" else 0
#     gender_other = 1 if gender == "Other" else 0
#     gender_female = 1 if gender == "Female" else 0
#     married_yes = 1 if married == "Yes" else 0
#     work_type_Never_worked = 1 if work == "Never_worked" else 0
#     work_type_Private = 1 if work == "Private" else 0
#     work_type_Self_employed = 1 if work == "Self-employed" else 0
#     work_type_children = 1 if work == "children" else 0
#     Residence_type_Urban = 1 if residence == "Urban" else 0
#     smoking_status_formerly_smoked = 1 if smoking == 'formerly smoked' else 0
#     smoking_status_never_smoked = 1 if smoking == 'never smoked' else 0
#     smoking_status_smokes = 1 if smoking == 'smokes' else 0
#
#     raw_features = [age, hypertension, disease, glucose, bmi,
#                     gender_male, gender_other, gender_female,
#                     married_yes,
#                     work_type_Never_worked, work_type_Private,
#                     work_type_Self_employed, work_type_children,
#                     Residence_type_Urban,
#                     smoking_status_formerly_smoked, smoking_status_never_smoked,
#                     smoking_status_smokes]
#
#     return scaler.transform([raw_features])
#
#
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == "POST":
#         feature = preprocess_input(request.form)
#         prediction = model.predict(feature)[0]
#         prediction = "YES" if prediction == 1 else "NO"
#         return render_template("index.html", prediction=prediction)
#
#     return render_template("index.html")
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open("model_pickle.pkl", 'rb'))
scaler = StandardScaler()

app = Flask(__name__)

# لیست ویژگی‌ها برای فرم (باید با index.html هماهنگ باشه)
features = [
    'age', 'hypertension', 'disease', 'glucose', 'bmi',
    'gender', 'married', 'work', 'residence', 'smoking'
]


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        form_data = {feature: request.form[feature] for feature in features}

        # پردازش داده‌ها
        age = int(form_data['age'])
        hypertension = int(form_data['hypertension'])
        disease = int(form_data['disease'])
        glucose = float(form_data['glucose'])
        bmi = float(form_data['bmi'])

        # gender
        gender = form_data['gender']
        gender_male = int(gender == "Male")
        gender_other = int(gender == "Other")
        gender_female = int(gender == "Female")

        # married
        married_yes = int(form_data['married'] == "Yes")

        # work type
        work = form_data['work']
        work_type_Never_worked = int(work == "Never_worked")
        work_type_Private = int(work == "Private")
        work_type_Self_employed = int(work == "Self-employed")
        work_type_children = int(work == "children")

        # residence type
        Residence_type_Urban = int(form_data['residence'] == "Urban")

        # smoking status
        smoking = form_data['smoking']
        smoking_status_formerly_smoked = int(smoking == 'formerly smoked')
        smoking_status_never_smoked = int(smoking == 'never smoked')
        smoking_status_smokes = int(smoking == 'smokes')

        final_features = np.array([[
            age, hypertension, disease, glucose, bmi,
            gender_male, gender_other, gender_female,
            married_yes,
            work_type_Never_worked, work_type_Private, work_type_Self_employed, work_type_children,
            Residence_type_Urban,
            smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes
        ]])

        final_features_scaled = scaler.fit_transform(final_features)
        prediction = model.predict(final_features_scaled)[0]
        prediction = "YES" if prediction == 1 else "NO"

        return render_template("index.html", features=features,
                               prediction_text=f"Chance of Stroke Prediction is --> {prediction}")

    return render_template("index.html", features=features)


if __name__ == "__main__":
    app.run(debug=True)
