from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model/fraud_model.pkl", "rb"))
model_columns = pickle.load(open("model/model_columns.pkl", "rb"))

THRESHOLD = 0.30

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect raw input
        input_dict = {
            "months_as_customer": float(request.form.get("months_as_customer", 0)),
            "total_claim_amount": float(request.form.get("total_claim_amount", 0)),
            "policy_deductable": float(request.form.get("policy_deductable", 0)),
            "policy_annual_premium": float(request.form.get("policy_annual_premium", 0)),
            "capital_gains": float(request.form.get("capital_gains", 0)),
            "capital_loss": float(request.form.get("capital_loss", 0)),
            "bodily_injuries": float(request.form.get("bodily_injuries", 0)),
            "witnesses": float(request.form.get("witnesses", 0)),
            "number_of_vehicles_involved": float(request.form.get("number_of_vehicles_involved", 1)),
            "incident_hour_of_the_day": float(request.form.get("incident_hour_of_the_day", 0)),
            "auto_year": float(request.form.get("auto_year", 2000)),
        }

        # Create empty dataframe with model columns
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0

        # Insert user values
        for key in input_dict:
            if key in input_df.columns:
                input_df.at[0, key] = input_dict[key]

        # Predict
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability > THRESHOLD else 0

        # Risk logic
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)