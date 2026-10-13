"""
Students' Academic Performance Prediction System
Using the Decision Tree Algorithm (CART, scikit-learn)

Features:
- Two prediction modes: Full-Term (with G2) and Early-Warning (before G2)
- Decision path explanation for every prediction
- Batch prediction from a class CSV, scored against actual results when the
  file includes the final grade (G3)
- Searchable, filterable and exportable prediction history (SQLite)
- Project details read from project.toml, a /health check, and a source-code download
"""
from flask import Flask, render_template, request, Response, send_file
import sqlite3
import os
import io
import csv
import json
import zipfile
import tomllib
import pandas as pd
from datetime import datetime

from predictor import (
    APP_DIR, FULL_FEATURES, EARLY_FEATURES, full_model, early_model,
    check_value, predict_one, explain_decision_path, read_student_csv,
    predict_class, score,
)

DB_PATH = os.environ.get("DB_PATH", os.path.join(APP_DIR, "predictions.db"))
RESULTS_PATH = os.path.join(APP_DIR, "model_results.json")
SAMPLE_PATH = os.path.join(APP_DIR, "sample_data", "student-por.csv")

STUDY_LABELS = ["<2h", "2-5h", "5-10h", ">10h"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit

with open(RESULTS_PATH) as f:
    model_results = json.load(f)

# Results of evaluate_new_data.py (the models tested on 649 unseen records)
NEW_DATA_PATH = os.path.join(APP_DIR, "new_data_results.json")
new_data_results = None
if os.path.exists(NEW_DATA_PATH):
    with open(NEW_DATA_PATH) as f:
        new_data_results = json.load(f)

with open(os.path.join(APP_DIR, "project.toml"), "rb") as f:
    project = tomllib.load(f)


@app.context_processor
def inject_project():
    return {"project": project}


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                prediction_mode TEXT DEFAULT 'Full Term',
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


init_db()


def save_predictions(records):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """INSERT INTO predictions
               (student_name, prediction_mode, studytime, failures, absences, g1, g2,
                predicted_category, top_factor, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [r + (now,) for r in records],
        )


@app.route("/")
def home():
    return render_template("home.html", results=model_results, new_data=new_data_results)


@app.route("/health")
def health():
    return Response("ok", mimetype="text/plain")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict_form.html", mode=request.args.get("mode", "full"), form={})

    mode = request.form.get("mode", "full")
    student_name = request.form.get("student_name", "").strip()[:100] or "Unnamed Student"
    features = EARLY_FEATURES if mode == "early" else FULL_FEATURES
    form_keys = {"studytime": "studytime", "failures": "failures",
                 "absences": "absences", "G1": "g1", "G2": "g2"}

    try:
        values = [check_value(f, request.form.get(form_keys[f], "")) for f in features]
    except ValueError as err:
        return render_template("predict_form.html", mode=mode, error=str(err), form=request.form)

    input_df = pd.DataFrame([values], columns=features)
    model = early_model if mode == "early" else full_model
    prediction, proba_map = predict_one(model, features, input_df)
    decision_rules, top_factor, advice = explain_decision_path(model, features, input_df)

    v = dict(zip(features, values))
    mode_display = "Early-Warning (CA Phase)" if mode == "early" else "Full-Term"
    inputs = {
        "Study Time (1-4 scale)": f"{v['studytime']} ({STUDY_LABELS[v['studytime'] - 1]})",
        "Past Class Failures": v["failures"],
        "Class Absences": v["absences"],
        "First Period Grade (G1)": f"{v['G1']} / 20",
    }
    if "G2" in v:
        inputs["Second Period Grade (G2)"] = f"{v['G2']} / 20"

    save_predictions([(student_name, mode_display, v["studytime"], v["failures"], v["absences"],
                       v["G1"], v.get("G2", -1), prediction, top_factor)])

    return render_template(
        "result.html", student_name=student_name, prediction=prediction,
        proba_map=proba_map, top_factor=top_factor, decision_rules=decision_rules,
        advice=advice, mode_display=mode_display, inputs=inputs,
    )


@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():
    if request.method == "GET":
        return render_template("batch_predict.html")

    if request.form.get("sample"):
        source, filename = SAMPLE_PATH, "student-por.csv (sample new data)"
    else:
        source = request.files.get("file")
        if not source or source.filename == "":
            return render_template("batch_predict.html", error="Please choose a CSV file to upload.")
        filename = source.filename

    try:
        df = read_student_csv(source)
        if len(df) > 5000:
            raise ValueError("The file has more than 5,000 rows. Please split it into smaller files.")
        rows, skipped, mode = predict_class(df)
    except ValueError as err:
        return render_template("batch_predict.html", error=str(err))
    except Exception:
        return render_template("batch_predict.html",
                               error="The file could not be read as a CSV. Please check its format.")

    if not rows:
        return render_template("batch_predict.html", error="No valid rows were found in the file.",
                               skipped=skipped)

    save_predictions([(r["name"], f"Batch ({mode})", r["studytime"], r["failures"], r["absences"],
                       r["G1"], r.get("G2", -1), r["predicted"], r["top_factor"]) for r in rows])

    counts = {c: sum(r["predicted"] == c for r in rows) for c in ("High", "Medium", "Low")}
    return render_template(
        "batch_predict.html", results_list=rows, counts=counts, total=len(rows),
        mode_str=mode, skipped=skipped, evaluation=score(rows), filename=filename,
    )


@app.route("/sample-data/<name>")
def sample_data(name):
    if name not in {"student-por.csv", "class_template.csv"}:
        return Response("Not found", status=404)
    return send_file(os.path.join(APP_DIR, "sample_data", name), as_attachment=True)


@app.route("/history")
def history():
    q = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if q:
        query += " AND student_name LIKE ?"
        params.append(f"%{q}%")
    if category in ["Low", "Medium", "High"]:
        query += " AND predicted_category = ?"
        params.append(category)
    query += " ORDER BY id DESC LIMIT 100"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return render_template("history.html", rows=rows, q=q, category=category)


def csv_safe(value):
    """Stops spreadsheet apps from running a name like '=SUM(...)' as a formula."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


@app.route("/history/export")
def export_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Student Name", "Prediction Mode", "Study Time", "Failures", "Absences",
                     "G1", "G2", "Predicted Category", "Top Factor", "Date"])
    for r in rows:
        writer.writerow([
            r["id"], csv_safe(r["student_name"]), r["prediction_mode"], r["studytime"],
            r["failures"], r["absences"], r["g1"], r["g2"] if r["g2"] != -1 else "N/A",
            r["predicted_category"], r["top_factor"], r["created_at"],
        ])

    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_history.csv"},
    )


@app.route("/about")
def about():
    return render_template("about.html", results=model_results, new_data=new_data_results)


SOURCE_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "screenshots"}


@app.route("/download/source")
def download_source():
    """Zips the project's source code, data and trained models for download."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(APP_DIR):
            dirs[:] = sorted(d for d in dirs if d not in SOURCE_SKIP_DIRS)
            for name in sorted(files):
                if name.endswith((".pyc", ".db")) or name == ".DS_Store":
                    continue
                path = os.path.join(root, name)
                arcname = os.path.join("academic-performance-system", os.path.relpath(path, APP_DIR))
                zf.write(path, arcname)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/zip", as_attachment=True,
                     download_name="academic-performance-system-source.zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
