"""
train_model.py
Trains CART Decision Tree classifiers on the UCI Student Performance dataset
(Cortez & Silva, 2008) to predict student academic performance categories.

Models trained:
1. Balanced Full-Term Model (decision_tree_model.joblib):
   Features: studytime, failures, absences, G1, G2.
   Regularized with max_features and balanced sampling to ensure study habits,
   failures, and attendance actively influence predictions rather than G2 dominating.
2. Early-Warning Model (early_warning_model.joblib):
   Features: studytime, failures, absences, G1 (Continuous Assessment Phase).
   Predicts academic risk before second-period/final exams are written.

Target derivation (0-20 scale):
  Low    : G3 <  10  (Fail threshold)
  Medium : 10 <= G3 < 15
  High   : G3 >= 15
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

RANDOM_STATE = 42

# ---------- 1. Load & Clean Dataset ----------
df = pd.read_csv("student-mat.csv", sep=";")
df.columns = [c.strip('"') for c in df.columns]
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip('"')

print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")

df["G3"] = df["G3"].astype(int)

def categorize(g3):
    if g3 < 10:
        return "Low"
    elif g3 < 15:
        return "Medium"
    else:
        return "High"

df["performance_category"] = df["G3"].apply(categorize)
labels = ["High", "Low", "Medium"]

# ---------- 2. Model 1: Balanced Full-Term Model (G1, G2, Attendance, Habits) ----------
FULL_FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
X_full = df[FULL_FEATURES].astype(float)
y = df["performance_category"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Regularized with max_features=3 to prevent G2 from monopolizing every node split,
# giving studytime, failures, absences, and G1 active decision rules.
full_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    max_features=3,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
full_model.fit(X_train_f, y_train_f)

y_pred_f = full_model.predict(X_test_f)
acc_f = accuracy_score(y_test_f, y_pred_f)
train_acc_f = accuracy_score(y_train_f, full_model.predict(X_train_f))
prec_f = precision_score(y_test_f, y_pred_f, average="weighted", zero_division=0)
rec_f = recall_score(y_test_f, y_pred_f, average="weighted", zero_division=0)
f1_f = f1_score(y_test_f, y_pred_f, average="weighted", zero_division=0)
cm_f = confusion_matrix(y_test_f, y_pred_f, labels=labels)
report_f = classification_report(y_test_f, y_pred_f, zero_division=0)

importances_f = dict(zip(FULL_FEATURES, full_model.feature_importances_))
importances_f_sorted = dict(sorted(importances_f.items(), key=lambda x: -x[1]))

print("\n=== BALANCED FULL-TERM MODEL ===")
print(f"Accuracy:  {acc_f:.4f} (training set: {train_acc_f:.4f})")
print(f"Precision: {prec_f:.4f}")
print(f"Recall:    {rec_f:.4f}")
print(f"F1-score:  {f1_f:.4f}")
print("Feature Importances:")
for k, v in importances_f_sorted.items():
    print(f"  {k}: {v*100:.1f}%")

# ---------- 3. Model 2: Early-Warning Model (Continuous Assessment Phase - No G2) ----------
EARLY_FEATURES = ["studytime", "failures", "absences", "G1"]
X_early = df[EARLY_FEATURES].astype(float)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_early, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

early_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
early_model.fit(X_train_e, y_train_e)

y_pred_e = early_model.predict(X_test_e)
acc_e = accuracy_score(y_test_e, y_pred_e)
train_acc_e = accuracy_score(y_train_e, early_model.predict(X_train_e))
prec_e = precision_score(y_test_e, y_pred_e, average="weighted", zero_division=0)
rec_e = recall_score(y_test_e, y_pred_e, average="weighted", zero_division=0)
f1_e = f1_score(y_test_e, y_pred_e, average="weighted", zero_division=0)
cm_e = confusion_matrix(y_test_e, y_pred_e, labels=labels)
report_e = classification_report(y_test_e, y_pred_e, zero_division=0)

importances_e = dict(zip(EARLY_FEATURES, early_model.feature_importances_))
importances_e_sorted = dict(sorted(importances_e.items(), key=lambda x: -x[1]))

print("\n=== EARLY WARNING MODEL (CA PHASE) ===")
print(f"Accuracy:  {acc_e:.4f} (training set: {train_acc_e:.4f})")
print(f"Precision: {prec_e:.4f}")
print(f"Recall:    {rec_e:.4f}")
print(f"F1-score:  {f1_e:.4f}")
print("Feature Importances:")
for k, v in importances_e_sorted.items():
    print(f"  {k}: {v*100:.1f}%")

# ---------- 4. Save Evaluation Results ----------
results = {
    "dataset": "UCI Student Performance (student-mat.csv), Cortez & Silva (2008)",
    "n_total": int(len(df)),
    "n_train": int(len(X_train_f)),
    "n_test": int(len(X_test_f)),
    "features": FULL_FEATURES,
    "early_features": EARLY_FEATURES,
    "classes": labels,
    "class_distribution": y.value_counts().to_dict(),
    "accuracy": round(acc_f, 4),
    "train_accuracy": round(train_acc_f, 4),
    "precision_weighted": round(prec_f, 4),
    "recall_weighted": round(rec_f, 4),
    "f1_weighted": round(f1_f, 4),
    "confusion_matrix": cm_f.tolist(),
    "confusion_matrix_labels": labels,
    "feature_importances": {k: round(v, 4) for k, v in importances_f_sorted.items()},
    "classification_report_text": report_f,
    "early_model": {
        "accuracy": round(acc_e, 4),
        "train_accuracy": round(train_acc_e, 4),
        "precision_weighted": round(prec_e, 4),
        "recall_weighted": round(rec_e, 4),
        "f1_weighted": round(f1_e, 4),
        "confusion_matrix": cm_e.tolist(),
        "feature_importances": {k: round(v, 4) for k, v in importances_e_sorted.items()},
        "classification_report_text": report_e
    },
    "model_params": {
        "criterion": "gini",
        "max_depth": 4,
        "max_features": 3,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    }
}

with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved evaluation results to model_results.json")

# ---------- 5. Visualize Decision Trees ----------
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    full_model, feature_names=FULL_FEATURES, class_names=full_model.classes_,
    filled=True, rounded=True, fontsize=10, ax=ax
)
ax.set_title("Balanced Decision Tree — Full Term Academic Prediction (max_depth=4, balanced)", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree_visualization.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(
    early_model, feature_names=EARLY_FEATURES, class_names=early_model.classes_,
    filled=True, rounded=True, fontsize=10, ax=ax
)
ax.set_title("Early-Warning Decision Tree — Continuous Assessment Phase (No G2)", fontsize=14)
plt.tight_layout()
plt.savefig("early_tree_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tree visualization diagrams.")

# ---------- 6. Feature Importance Chart ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

names_f = list(importances_f_sorted.keys())
vals_f = [v * 100 for v in importances_f_sorted.values()]
ax1.barh(names_f[::-1], vals_f[::-1], color="#2c7fb8")
ax1.set_xlabel("Importance (%)")
ax1.set_title("Balanced Full-Term Model (With G2)")

names_e = list(importances_e_sorted.keys())
vals_e = [v * 100 for v in importances_e_sorted.values()]
ax2.barh(names_e[::-1], vals_e[::-1], color="#2e7d5b")
ax2.set_xlabel("Importance (%)")
ax2.set_title("Early-Warning Model (CA Phase - Without G2)")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved feature_importance.png")

# ---------- 7. Confusion Matrix Chart ----------
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm_f, cmap="Blues")
ax.set_title("Confusion Matrix — Balanced Test Set")
fig.colorbar(ax.images[0], ax=ax)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, str(cm_f[i, j]), ha="center", va="center",
                color="white" if cm_f[i, j] > cm_f.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved confusion_matrix.png")

# ---------- 8. Save Models ----------
joblib.dump(full_model, "decision_tree_model.joblib")
joblib.dump(early_model, "early_warning_model.joblib")
print("Saved decision_tree_model.joblib and early_warning_model.joblib")

# ---------- 9. Save Readable Tree Rules ----------
rules_full = export_text(full_model, feature_names=FULL_FEATURES)
rules_early = export_text(early_model, feature_names=EARLY_FEATURES)
with open("tree_rules.txt", "w") as f:
    f.write("=== BALANCED FULL-TERM TREE RULES ===\n")
    f.write(rules_full)
    f.write("\n\n=== EARLY-WARNING TREE RULES (WITHOUT G2) ===\n")
    f.write(rules_early)
print("Saved tree_rules.txt")
print("\nTRAINING COMPLETE.")

