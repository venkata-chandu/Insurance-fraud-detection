# ------------------- IMPORTS -------------------
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import pickle
import sqlite3
import csv
import io
import os

# ------------------- CREATE APP -------------------
app = Flask(__name__)

# ------------------- LOAD MODEL -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "model_columns.pkl")

model = pickle.load(open(model_path, "rb"))
model_columns = pickle.load(open(columns_path, "rb"))

# ------------------- DATABASE INIT -------------------
def init_db():
    conn = sqlite3.connect("claims.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        months_as_customer REAL,
        total_claim_amount REAL,
        policy_deductable REAL,
        policy_annual_premium REAL,
        capital_gains REAL,
        capital_loss REAL,
        bodily_injuries REAL,
        witnesses REAL,
        number_of_vehicles_involved REAL,
        incident_hour REAL,
        auto_year REAL,
        probability REAL,
        prediction INTEGER,
        risk_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ------------------- SAVE FUNCTION -------------------
def save_to_db(data, probability, prediction, risk_level):
    conn = sqlite3.connect("claims.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO claims (
        months_as_customer,
        total_claim_amount,
        policy_deductable,
        policy_annual_premium,
        capital_gains,
        capital_loss,
        bodily_injuries,
        witnesses,
        number_of_vehicles_involved,
        incident_hour,
        auto_year,
        probability,
        prediction,
        risk_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["months_as_customer"],
        data["total_claim_amount"],
        data["policy_deductable"],
        data["policy_annual_premium"],
        data["capital_gains"],
        data["capital_loss"],
        data["bodily_injuries"],
        data["witnesses"],
        data["number_of_vehicles_involved"],
        data["incident_hour_of_the_day"],
        data["auto_year"],
        probability,
        prediction,
        risk_level
    ))

    conn.commit()
    conn.close()

# ------------------- ROUTES -------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        def get_float(field, default=0):
            val = request.form.get(field, default)
            return float(val) if val not in ["", None] else default

        input_dict = {
            "months_as_customer": get_float("months_as_customer"),
            "total_claim_amount": get_float("total_claim_amount"),
            "policy_deductable": get_float("policy_deductable"),
            "policy_annual_premium": get_float("policy_annual_premium"),
            "capital_gains": get_float("capital_gains"),
            "capital_loss": get_float("capital_loss"),
            "bodily_injuries": get_float("bodily_injuries"),
            "witnesses": get_float("witnesses"),
            "number_of_vehicles_involved": get_float("number_of_vehicles_involved", 1),
            "incident_hour_of_the_day": get_float("incident_hour_of_the_day"),
            "auto_year": get_float("auto_year", 2000),
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        probability = float(model.predict_proba(input_df)[0][1])

        THRESHOLD = 0.30
        prediction = 1 if probability > THRESHOLD else 0

        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        save_to_db(input_dict, probability, prediction, risk_level)

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/export")
def export_csv():
    conn = sqlite3.connect("claims.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM claims")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(columns)
    writer.writerows(rows)

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="claims_data.csv"
    )

# ------------------- RUN -------------------
if __name__ == "__main__":
    app.run(debug=True)