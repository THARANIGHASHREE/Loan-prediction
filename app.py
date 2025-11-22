from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# -------------------------------
# Load Model & Encoders
# -------------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# -------------------------------
# Utility: Encode Categorical Input
# -------------------------------
def encode_value(col_name, value):
    """Encodes a categorical value using the saved label encoders."""
    if col_name in encoders:
        return encoders[col_name].transform([value])[0]
    return value

# -------------------------------
# Home Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Collect Input Values
    gender = request.form["gender"]
    married = request.form["married"]
    dependents = request.form.get("dependents", "0")  # optional
    education = request.form["education"]
    self_employed = request.form.get("self_employed", "No")  # optional
    property_area = request.form["property_area"]

    applicant_income = float(request.form["applicant_income"])
    coapplicant_income = float(request.form["coapplicant_income"])
    loan_amount = float(request.form["loan_amount"])
    loan_term = float(request.form["loan_term"])
    credit_history = int(request.form["credit_history"])

    # -------------------------------
    # Encode Categorical Variables
    # -------------------------------
    gender = encode_value("Gender", gender)
    married = encode_value("Married", married)
    dependents = encode_value("Dependents", dependents)
    education = encode_value("Education", education)
    self_employed = encode_value("Self_Employed", self_employed)
    property_area = encode_value("Property_Area", property_area)

    # -------------------------------
    # Build Input Vector
    # Order MUST match training order
    # -------------------------------
    input_data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    # -------------------------------
    # Predict
    # -------------------------------
    prediction = model.predict(input_data)[0]

    result = "Loan Approved ✔" if prediction == 1 else "Loan Not Approved ❌"

    return render_template("index.html", result=result)

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
