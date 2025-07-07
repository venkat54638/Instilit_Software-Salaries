from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# Load trained pipeline with preprocessing inside
pipeline = joblib.load("saved_models/final_XGBoost_pipelinenew.pkl")

# PostgreSQL DB connection
DB_URI = 'postgresql+psycopg2://postgres:1234@localhost:5432/salary_db'
engine = create_engine(DB_URI)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_text = None

    if request.method == "POST":
        try:
            # Extract only required input features
            input_data = {
                'job_title': request.form.get('job_title', '').lower().strip(),
                'experience_level': request.form.get('experience_level', ''),
                'employment_type': request.form.get('employment_type', ''),
                'company_size': request.form.get('company_size', ''),
                'company_location': request.form.get('company_location', ''),
                'remote_ratio': float(request.form.get('remote_ratio', 0)),
                'years_experience': float(request.form.get('years_experience', 0)),
                'salary_currency': request.form.get('salary_currency', ''),
                'conversion_rate': float(request.form.get('conversion_rate', 1))
            }

            input_df = pd.DataFrame([input_data])
            log_salary = pipeline.predict(input_df)[0]
            predicted_usd_salary = np.expm1(log_salary)
            converted_salary = predicted_usd_salary * input_data['conversion_rate']
            currency_name = input_data['salary_currency'].upper()

            prediction_text = f"💰 Predicted Adjusted Salary: {converted_salary:,.2f} {currency_name}"

            # Add prediction result for logging
            input_data['adjusted_total_usd'] = round(predicted_usd_salary, 2)
            pd.DataFrame([input_data]).to_sql("salarytbl", engine, if_exists="append", index=False)

        except Exception as e:
            prediction_text = f"❌ Error: {str(e)}"

    return render_template("predict.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
