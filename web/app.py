from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load("../models/bst_diabetes_classifier.pkl")

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('main.html')

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=["POST"])
def predict():

    # get data from form
    age = int(request.form["age"])
    gender = request.form["gender"]
    bmi = float(request.form["bmi"])
    chol = float(request.form["chol"])
    tg = float(request.form["tg"])
    hdl = float(request.form["hdl"])
    ldl = float(request.form["ldl"])
    cr = float(request.form["cr"])
    bun = float(request.form["bun"])

    gender_encoded = 1 if gender == "M" else 0 # Male is encoded as 1 and Female as 0

    # store data as numpy array
    X = np.array([[
        age,
        gender_encoded,
        bmi,
        chol,
        tg,
        hdl,
        ldl,
        cr,
        bun
    ]])

    # 🔹 prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return render_template(
        "results.html",
        prediction=int(prediction),
        probability=round(probability * 100, 2)
    )