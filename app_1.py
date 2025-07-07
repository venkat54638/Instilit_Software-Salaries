from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'

# Load the trained pipeline (includes preprocessing + model)
pipeline = joblib.load("saved_models/final_XGBoost_pipelinenew.pkl")

# PostgreSQL DB connection
DB_URI = 'postgresql+psycopg2://postgres:1234@localhost:5432/salary_db'
engine = create_engine(DB_URI)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file:
            return render_template("index.html", error="❌ Please upload a CSV file.")

        try:
            df = pd.read_csv(file)

            # Expected columns
            required_cols = [
                'job_title', 'experience_level', 'employment_type',
                'company_size', 'company_location', 'remote_ratio',
                'years_experience', 'salary_currency', 'conversion_rate'
            ]

            if not all(col in df.columns for col in required_cols):
                missing = list(set(required_cols) - set(df.columns))
                return render_template("index.html", error=f"❌ Missing columns: {missing}")

            # Predict
            predictions_log = pipeline.predict(df[required_cols])
            df["predicted_log_salary"] = predictions_log
            df["adjusted_total_usd"] = np.expm1(predictions_log)
            df["converted_salary"] = df["adjusted_total_usd"] * df["conversion_rate"]

            #  Save to PostgreSQL
            df_to_store = df[required_cols + ['adjusted_total_usd', 'converted_salary']]
            df_to_store.to_sql("salarytbl", engine, if_exists="append", index=False)

            # Render HTML table
            result_html = df_to_store.to_html(classes="table table-striped", index=False)
            return render_template("predict.html", table=result_html)

        except Exception as e:
            return render_template("index.html", error=f"❌ Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
