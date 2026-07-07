from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("credit_card_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "CODE_GENDER": [request.form["gender"]],
        "FLAG_OWN_CAR": [request.form["car"]],
        "FLAG_OWN_REALTY": [request.form["realty"]],
        "CNT_CHILDREN": [int(request.form["children"])],
        "AMT_INCOME_TOTAL": [float(request.form["income"])],
        "NAME_INCOME_TYPE": [request.form["income_type"]],
        "NAME_EDUCATION_TYPE": [request.form["education"]],
        "NAME_FAMILY_STATUS": [request.form["family_status"]],
        "NAME_HOUSING_TYPE": [request.form["housing"]],
        "DAYS_BIRTH": [int(request.form["days_birth"])],
        "DAYS_EMPLOYED": [int(request.form["days_employed"])],
        "FLAG_MOBIL": [1],
        "FLAG_WORK_PHONE": [int(request.form["work_phone"])],
        "FLAG_PHONE": [int(request.form["phone"])],
        "FLAG_EMAIL": [int(request.form["email"])],
        "OCCUPATION_TYPE": [request.form["occupation"]],
        "CNT_FAM_MEMBERS": [float(request.form["family_members"])]
    }

    df = pd.DataFrame(data)

    prob = model.predict_proba(df)[0]
    approved = prob[0]
    rejected = prob[1]
    print("Approved Probability:", approved)
    print("Rejected Probability:", rejected)
    # Lower threshold for rejection
    if rejected >= 0.25:
        result = f"❌ Credit Card Rejected ({rejected*100:.2f}%)"
    else:
        result = f"✅ Credit Card Approved ({approved*100:.2f}%)"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)