"""
Students' Academic Performance Prediction System
Using the Decision Tree Algorithm (CART, scikit-learn)

Features:
- Dual-Stage Decision Tree Prediction (Early Warning CA Phase vs. Full-Term Comprehensive)
- Dynamic Decision-Tree Path Traversal & Explainability Engine
- Batch Prediction via CSV Upload & Export
- Searchable, Filterable, and Exportable History Log (SQLite)
- Robust server-side validation
"""
from flask import Flask, render_template, request, redirect, url_for, Response, send_file
import joblib
import sqlite3
import os
import io
import csv
import json
import pandas as pd
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "decision_tree_model.joblib")
EARLY_MODEL_PATH = os.path.join(APP_DIR, "early_warning_model.joblib")
DB_PATH = os.path.join(APP_DIR, "predictions.db")
RESULTS_PATH = os.path.join(APP_DIR, "model_results.json")

FULL_FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
EARLY_FEATURES = ["studytime", "failures", "absences", "G1"]

app = Flask(__name__)

# Load trained models & metadata
full_model = joblib.load(MODEL_PATH)
early_model = joblib.load(EARLY_MODEL_PATH) if os.path.exists(EARLY_MODEL_PATH) else full_model

with open(RESULTS_PATH) as f:
    model_results = json.load(f)


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
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(predictions)")
        existing_cols = [c[1] for c in cur.fetchall()]
        if "prediction_mode" not in existing_cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN prediction_mode TEXT DEFAULT 'Full Term'")
        conn.commit()


init_db()


def explain_decision_path(model, feature_names, input_df):
    """
    Traces the exact nodes traversed in the CART Decision Tree for this specific student,
    extracting the exact split conditions, features evaluated, and generating
    an intuitive explanation of why the model reached this classification.
    """
    node_indicator = model.decision_path(input_df)
    leaf_id = model.apply(input_df)[0]
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    rules = []
    features_hit = set()
    feature_labels = {
        "studytime": "Weekly Study Time",
        "failures": "Past Failures",
        "absences": "Class Absences",
        "G1": "First Period Grade (G1)",
        "G2": "Second Period Grade (G2)",
    }

    for node_id in node_index:
        if leaf_id == node_id:
            continue

        feat_idx = feature[node_id]
        feat_name = feature_names[feat_idx]
        features_hit.add(feat_name)
        thresh = threshold[node_id]
        val = input_df[feat_name].values[0]

        if val <= thresh:
            op = "<="
            desc = f"{feature_labels.get(feat_name, feat_name)} is {val} (which is ≤ {thresh:.1f})"
        else:
            op = ">"
            desc = f"{feature_labels.get(feat_name, feat_name)} is {val} (which is > {thresh:.1f})"

        rules.append({
            "feature": feat_name,
            "feature_label": feature_labels.get(feat_name, feat_name),
            "operator": op,
            "threshold": round(float(thresh), 1),
            "value": val,
            "description": desc
        })

    importances = dict(zip(feature_names, model.feature_importances_))
    active_importances = {f: importances[f] for f in features_hit if f in importances}
    top_factor = max(active_importances, key=active_importances.get) if active_importances else feature_names[0]
    top_factor_label = feature_labels.get(top_factor, top_factor)

    advice = []
    val_map = input_df.iloc[0].to_dict()
    if val_map.get("absences", 0) > 8:
        advice.append("Attendance Risk: High absences detected. Direct attendance counseling recommended.")
    if val_map.get("studytime", 2) <= 1:
        advice.append("Study Habit: Weekly study time is under 2 hours. Increased independent study hours recommended.")
    if val_map.get("failures", 0) > 0:
        advice.append("Remediation: Student has past course failures. Targeted tutorial sessions recommended.")
    if val_map.get("G1", 10) < 10:
        advice.append("Continuous Assessment Warning: First-period score is below the pass mark (10/20). Early academic intervention advised.")

    return rules, top_factor_label, advice


@app.route("/")
def home():
    return render_template("home.html", results=model_results)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        mode = request.args.get("mode", "full")
        return render_template("predict_form.html", mode=mode)

    # Form Submission
    mode = request.form.get("mode", "full")
    student_name = request.form.get("student_name", "").strip() or "Unnamed Student"

    try:
        studytime = int(request.form["studytime"])
        failures = int(request.form["failures"])
        absences = int(request.form["absences"])
        g1 = int(request.form["g1"])

        if not (1 <= studytime <= 4):
            raise ValueError("Study time must be between 1 and 4.")
        if not (0 <= failures <= 10):
            raise ValueError("Failures must be non-negative.")
        if not (0 <= absences <= 100):
            raise ValueError("Absences must be between 0 and 100.")
        if not (0 <= g1 <= 20):
            raise ValueError("G1 grade must be between 0 and 20.")

        if mode == "early":
            g2 = None
            active_model = early_model
            active_features = EARLY_FEATURES
            input_df = pd.DataFrame([[studytime, failures, absences, g1]], columns=EARLY_FEATURES)
            mode_display = "Early-Warning (CA Phase)"
            input_dict = {
                "Study Time (1-4 scale)": f"{studytime} ({['<2h', '2-5h', '5-10h', '>10h'][studytime-1]})",
                "Past Class Failures": failures,
                "Class Absences": absences,
                "First Period Grade (G1)": f"{g1} / 20",
            }
        else:
            g2 = int(request.form["g2"])
            if not (0 <= g2 <= 20):
                raise ValueError("G2 grade must be between 0 and 20.")
            active_model = full_model
            active_features = FULL_FEATURES
            input_df = pd.DataFrame([[studytime, failures, absences, g1, g2]], columns=FULL_FEATURES)
            mode_display = "Comprehensive Full-Term"
            input_dict = {
                "Study Time (1-4 scale)": f"{studytime} ({['<2h', '2-5h', '5-10h', '>10h'][studytime-1]})",
                "Past Class Failures": failures,
                "Class Absences": absences,
                "First Period Grade (G1)": f"{g1} / 20",
                "Second Period Grade (G2)": f"{g2} / 20",
            }

    except (KeyError, ValueError) as err:
        error_msg = str(err) if str(err) else "Please fill in all required fields with valid values."
        return render_template("predict_form.html", mode=mode, error=error_msg)

    # Perform inference
    prediction = active_model.predict(input_df)[0]
    proba = active_model.predict_proba(input_df)[0]
    classes = list(active_model.classes_)
    proba_map = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    # Dynamic explainability & decision path
    decision_rules, top_factor, advice = explain_decision_path(active_model, active_features, input_df)

    # Save to SQLite database
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO predictions
               (student_name, prediction_mode, studytime, failures, absences, g1, g2,
                predicted_category, top_factor, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (student_name, mode_display, studytime, failures, absences, g1, g2 if g2 is not None else -1,
             prediction, top_factor, datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
        conn.commit()

    return render_template(
        "result.html",
        student_name=student_name,
        prediction=prediction,
        proba_map=proba_map,
        top_factor=top_factor,
        decision_rules=decision_rules,
        advice=advice,
        mode_display=mode_display,
        inputs=input_dict
    )


@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():
    if request.method == "GET":
        return render_template("batch_predict.html")

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("batch_predict.html", error="Please select a CSV file to upload.")

    try:
        df_batch = pd.read_csv(file)
        # Standardize column headers (case-insensitive strip)
        col_map = {str(c).strip().lower(): c for c in df_batch.columns}
        
        # Check required columns
        req_cols = ["studytime", "failures", "absences", "g1"]
        for rc in req_cols:
            if rc not in col_map:
                return render_template("batch_predict.html", error=f"Missing required column in CSV: '{rc}'")

        has_g2 = "g2" in col_map
        active_model = full_model if has_g2 else early_model
        active_features = FULL_FEATURES if has_g2 else EARLY_FEATURES
        mode_str = "Full Term" if has_g2 else "Early Warning"

        results_list = []
        counts = {"High": 0, "Medium": 0, "Low": 0}

        for idx, row in df_batch.iterrows():
            name = str(row[col_map.get("name", col_map.get("student_name", list(df_batch.columns)[0]))]) if "name" in col_map or "student_name" in col_map else f"Student #{idx+1}"
            studytime = int(row[col_map["studytime"]])
            failures = int(row[col_map["failures"]])
            absences = int(row[col_map["absences"]])
            g1 = int(row[col_map["g1"]])
            g2 = int(row[col_map["g2"]]) if has_g2 else None

            if has_g2:
                feat_df = pd.DataFrame([[studytime, failures, absences, g1, g2]], columns=FULL_FEATURES)
            else:
                feat_df = pd.DataFrame([[studytime, failures, absences, g1]], columns=EARLY_FEATURES)

            pred = active_model.predict(feat_df)[0]
            counts[pred] = counts.get(pred, 0) + 1

            results_list.append({
                "name": name,
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "g1": g1,
                "g2": g2 if g2 is not None else "N/A",
                "predicted": pred,
            })

            # Save batch item to DB
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO predictions
                       (student_name, prediction_mode, studytime, failures, absences, g1, g2,
                        predicted_category, top_factor, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, f"Batch ({mode_str})", studytime, failures, absences, g1, g2 if g2 is not None else -1,
                     pred, "Batch Import", datetime.now().strftime("%Y-%m-%d %H:%M"))
                )

        return render_template(
            "batch_predict.html",
            results_list=results_list,
            counts=counts,
            total=len(results_list),
            mode_str=mode_str
        )

    except Exception as e:
        return render_template("batch_predict.html", error=f"Error parsing CSV: {str(e)}")


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


@app.route("/history/export")
def export_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Student Name", "Prediction Mode", "Study Time", "Failures", "Absences", "G1", "G2", "Predicted Category", "Top Factor", "Date"])

    for r in rows:
        writer.writerow([
            r["id"],
            r["student_name"],
            r["prediction_mode"] if "prediction_mode" in r.keys() else "Full Term",
            r["studytime"],
            r["failures"],
            r["absences"],
            r["g1"],
            r["g2"] if r["g2"] != -1 else "N/A",
            r["predicted_category"],
            r["top_factor"],
            r["created_at"]
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_history.csv"}
    )


@app.route("/about")
def about():
    return render_template("about.html", results=model_results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

