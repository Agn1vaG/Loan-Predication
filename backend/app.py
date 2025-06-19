from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("loan_model.pkl", "rb"))

# Home page route - shows the form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction route - processes form input
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    # Convert form inputs to a list of values in the correct order
    input_data = [
        int(data["Gender"]),
        int(data["Married"]),
        int(data["Dependents"]),
        int(data["Education"]),
        int(data["Self_Employed"]),
        float(data["ApplicantIncome"]),
        float(data["CoapplicantIncome"]),
        float(data["LoanAmount"]),
        float(data["Loan_Amount_Term"]),
        float(data["Credit_History"]),
        int(data["Property_Area"])
    ]

    # Make prediction
    prediction = model.predict([input_data])
    result = "Approved" if prediction[0] == 1 else "Rejected"

    return f"<h2>Loan {result}</h2>"

# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)
