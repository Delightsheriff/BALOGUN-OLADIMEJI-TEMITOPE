"""
Students' Academic Performance Prediction System
Using the Decision Tree Algorithm (CART, scikit-learn)

A lightweight Flask application. Single form -> single prediction -> optional
history log. Deliberately minimal per project scope (see Chapter 1, Scope
and Limitations).
"""
from flask import Flask, render_template, request, redirect, url_for
import joblib
import sqlite3
import os
import json
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "decision_tree_model.joblib")
DB_PATH = os.path.join(APP_DIR, "predictions.db")
RESULTS_PATH = os.path.join(APP_DIR, "model_results.json")

FEATURES = ["studytime", "failures", "absences", "G1", "G2"]

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
with open(RESULTS_PATH) as f:
    model_results = json.load(f)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            studytime INTEGER,
            failures INTEGER,
            absences INTEGER,
            g1 INTEGER,
            g2 INTEGER,
            predicted_category TEXT,
            top_factor TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


@app.route("/")
def home():
    return render_template("home.html", results=model_results)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict_form.html")

    # ---- Read and validate form input ----
    try:
        student_name = request.form.get("student_name", "").strip() or "Unnamed Student"
        studytime = int(request.form["studytime"])
        failures = int(request.form["failures"])
        absences = int(request.form["absences"])
        g1 = int(request.form["g1"])
        g2 = int(request.form["g2"])
    except (KeyError, ValueError):
        return render_template(
            "predict_form.html",
            error="Please fill in all fields with valid numbers."
        )

    input_row = [[studytime, failures, absences, g1, g2]]
    prediction = model.predict(input_row)[0]
    proba = model.predict_proba(input_row)[0]
    classes = list(model.classes_)
    proba_map = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    # Top contributing factor: the feature with highest global importance
    # among the ones that actually appear in the path taken for this input.
    importances = model_results["feature_importances"]
    top_factor = max(importances, key=importances.get)

    # ---- Save to history (SQLite) ----
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (student_name, studytime, failures, absences, g1, g2,
            predicted_category, top_factor, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (student_name, studytime, failures, absences, g1, g2,
         prediction, top_factor, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        student_name=student_name,
        prediction=prediction,
        proba_map=proba_map,
        top_factor=top_factor,
        inputs={
            "Study Time (1-4 scale)": studytime,
            "Past Class Failures": failures,
            "Absences": absences,
            "First Period Grade (G1)": g1,
            "Second Period Grade (G2)": g2,
        }
    )


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


@app.route("/about")
def about():
    return render_template("about.html", results=model_results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
