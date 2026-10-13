"""
predictor.py — shared prediction logic used by the web app (app.py) and by
evaluate_new_data.py.

Keeps the two trained Decision Trees, the feature lists, the grade-to-category
rule and the CSV reading/validation in one place, so the app and the
evaluation script always treat data the same way the training script did.
"""
import os
import pandas as pd
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
EARLY_FEATURES = ["studytime", "failures", "absences", "G1"]
LABELS = ["High", "Low", "Medium"]

# Allowed input ranges (same limits as the prediction form)
RANGES = {
    "studytime": (1, 4),
    "failures": (0, 10),
    "absences": (0, 100),
    "G1": (0, 20),
    "G2": (0, 20),
}

FEATURE_LABELS = {
    "studytime": "Weekly Study Time",
    "failures": "Past Failures",
    "absences": "Class Absences",
    "G1": "First Period Grade (G1)",
    "G2": "Second Period Grade (G2)",
}

full_model = joblib.load(os.path.join(APP_DIR, "decision_tree_model.joblib"))
early_model = joblib.load(os.path.join(APP_DIR, "early_warning_model.joblib"))


def categorize(g3):
    """Final grade (0-20) to performance category, as in train_model.py."""
    if g3 < 10:
        return "Low"
    if g3 < 15:
        return "Medium"
    return "High"


def check_value(feature, value):
    """Returns the value as an int, or raises ValueError with a readable message."""
    label = FEATURE_LABELS[feature]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.")
    if pd.isna(number) or number != int(number):
        raise ValueError(f"{label} must be a whole number.")
    low, high = RANGES[feature]
    if not (low <= number <= high):
        raise ValueError(f"{label} must be between {low} and {high}.")
    return int(number)


def read_student_csv(file_obj):
    """
    Reads a class CSV. The delimiter is detected automatically, so both a
    simple comma-separated roster and the original semicolon-separated UCI
    files (student-mat.csv, student-por.csv) are accepted. Column names are
    matched without regard to case or surrounding quotes/spaces.
    """
    df = pd.read_csv(file_obj, sep=None, engine="python")
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    rename = {}
    for wanted in FULL_FEATURES + ["G3"]:
        if wanted.lower() in lookup:
            rename[lookup[wanted.lower()]] = wanted
    for name_col in ("student_name", "name"):
        if name_col in lookup:
            rename[lookup[name_col]] = "student_name"
            break
    return df.rename(columns=rename)


def predict_one(model, features, row_df):
    prediction = model.predict(row_df)[0]
    proba = model.predict_proba(row_df)[0]
    proba_map = {c: round(float(p) * 100, 1) for c, p in zip(model.classes_, proba)}
    return prediction, proba_map


def explain_decision_path(model, feature_names, input_df):
    """
    Traces the nodes this student passes through in the Decision Tree and
    returns the split conditions met, the most important feature on that
    path, and simple advising notes.
    """
    node_indicator = model.decision_path(input_df)
    leaf_id = model.apply(input_df)[0]
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    rules = []
    features_hit = set()
    for node_id in node_index:
        if leaf_id == node_id:
            continue
        feat_name = feature_names[feature[node_id]]
        features_hit.add(feat_name)
        thresh = threshold[node_id]
        val = input_df[feat_name].values[0]
        op = "<=" if val <= thresh else ">"
        sign = "≤" if op == "<=" else ">"
        label = FEATURE_LABELS.get(feat_name, feat_name)
        rules.append({
            "feature": feat_name,
            "feature_label": label,
            "operator": op,
            "threshold": round(float(thresh), 1),
            "value": val,
            "description": f"{label} is {val} (which is {sign} {thresh:.1f})",
        })

    importances = dict(zip(feature_names, model.feature_importances_))
    active = {f: importances[f] for f in features_hit if f in importances}
    top_factor = max(active, key=active.get) if active else feature_names[0]

    advice = []
    values = input_df.iloc[0].to_dict()
    if values.get("absences", 0) > 8:
        advice.append("Attendance: high number of absences. Attendance counselling is recommended.")
    if values.get("studytime", 2) <= 1:
        advice.append("Study habit: under 2 hours of study per week. More independent study is recommended.")
    if values.get("failures", 0) > 0:
        advice.append("Remediation: the student has failed courses before. Targeted tutorials are recommended.")
    if values.get("G1", 10) < 10:
        advice.append("Continuous assessment: the first-period grade is below the pass mark (10/20). Early intervention is advised.")

    return rules, FEATURE_LABELS.get(top_factor, top_factor), advice


def predict_class(df):
    """
    Predicts every row of a class DataFrame (already passed through
    read_student_csv). Uses the full-term model when a G2 column is present
    and the early-warning model otherwise.

    Returns (rows, skipped, mode). Each row holds the inputs, the prediction,
    the top factor and, when the file has a G3 column, the actual category.
    Rows with missing or out-of-range values are skipped and reported.
    """
    missing = [c for c in EARLY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    has_g2 = "G2" in df.columns
    model = full_model if has_g2 else early_model
    features = FULL_FEATURES if has_g2 else EARLY_FEATURES
    mode = "Full Term" if has_g2 else "Early Warning"
    has_g3 = "G3" in df.columns

    rows, skipped = [], []
    for i, record in df.iterrows():
        line = i + 2  # header is line 1
        try:
            values = [check_value(f, record[f]) for f in features]
        except ValueError as err:
            skipped.append(f"Row {line}: {err}")
            continue
        row_df = pd.DataFrame([values], columns=features)
        prediction = model.predict(row_df)[0]
        _, top_factor, _ = explain_decision_path(model, features, row_df)
        name = record.get("student_name") if "student_name" in df.columns else None
        row = dict(zip(features, values))
        row.update({
            "name": str(name).strip() if name is not None and not pd.isna(name) else f"Student #{i + 1}",
            "predicted": prediction,
            "top_factor": top_factor,
            "actual": None,
        })
        if has_g3:
            try:
                row["actual"] = categorize(check_value_grade(record["G3"]))
            except ValueError:
                row["actual"] = None
        rows.append(row)
    return rows, skipped, mode


def check_value_grade(value):
    number = float(value)
    if pd.isna(number) or not (0 <= number <= 20):
        raise ValueError("G3 out of range")
    return number


def score(rows):
    """Accuracy and confusion matrix for rows whose actual category is known."""
    labelled = [r for r in rows if r["actual"]]
    if not labelled:
        return None
    correct = sum(r["actual"] == r["predicted"] for r in labelled)
    matrix = {a: {p: 0 for p in LABELS} for a in LABELS}
    for r in labelled:
        matrix[r["actual"]][r["predicted"]] += 1
    return {
        "n": len(labelled),
        "correct": correct,
        "accuracy": round(correct / len(labelled) * 100, 1),
        "matrix": matrix,
    }
