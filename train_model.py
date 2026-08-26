"""
train_model.py
Trains a CART Decision Tree classifier on the UCI Student Performance dataset
(Cortez & Silva, 2008) to predict a student's academic performance category.

Target derivation: G3 (final grade, 0-20) is converted into a 3-class category:
  Low    : G3 <  10  (fail threshold per the original dataset documentation)
  Medium : 10 <= G3 < 15
  High   : G3 >= 15

Input features (deliberately minimal, matching Chapter 1's scope decision):
  studytime, failures, absences, G1, G2
These map directly onto CA-score / attendance / prior-grade style variables
that a lecturer would realistically have on hand.
"""
import pandas as pd
import numpy as np
import json
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

# ---------- 1. Load ----------
df = pd.read_csv("student-mat.csv", sep=";")
df.columns = [c.strip('"') for c in df.columns]
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip('"')

print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")

# ---------- 2. Feature selection & target derivation ----------
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
df["G3"] = df["G3"].astype(int)

def categorize(g3):
    if g3 < 10:
        return "Low"
    elif g3 < 15:
        return "Medium"
    else:
        return "High"

df["performance_category"] = df["G3"].apply(categorize)

X = df[FEATURES].astype(float)
y = df["performance_category"]

print("\nClass distribution:")
print(y.value_counts())

# ---------- 3. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---------- 4. Train CART Decision Tree (depth-limited to control overfitting) ----------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# ---------- 5. Evaluate ----------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\n--- Test Set Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

report = classification_report(y_test, y_pred, zero_division=0)
print("\nClassification Report:\n", report)

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion Matrix (rows=actual, cols=predicted), labels:", labels)
print(cm)

# ---------- 6. Feature importance ----------
importances = dict(zip(FEATURES, model.feature_importances_))
importances_sorted = dict(sorted(importances.items(), key=lambda x: -x[1]))
print("\nFeature importances:")
for k, v in importances_sorted.items():
    print(f"  {k}: {v:.4f}")

# ---------- 7. Save evaluation results for Chapter 4 ----------
results = {
    "dataset": "UCI Student Performance (student-mat.csv), Cortez & Silva (2008)",
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": FEATURES,
    "classes": labels,
    "class_distribution": y.value_counts().to_dict(),
    "accuracy": round(accuracy, 4),
    "precision_weighted": round(precision, 4),
    "recall_weighted": round(recall, 4),
    "f1_weighted": round(f1, 4),
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_labels": labels,
    "feature_importances": {k: round(v, 4) for k, v in importances_sorted.items()},
    "classification_report_text": report,
    "model_params": {
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE
    }
}
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved evaluation results to model_results.json")

# ---------- 8. Visualize the tree ----------
plt.figure(figsize=(20, 10))
plot_tree(
    model, feature_names=FEATURES, class_names=model.classes_,
    filled=True, rounded=True, fontsize=10, proportion=False
)
plt.title("Decision Tree — Student Academic Performance Prediction (max_depth=4, Gini)", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved decision_tree_visualization.png")

# ---------- 9. Feature importance chart ----------
plt.figure(figsize=(8, 5))
names = list(importances_sorted.keys())
vals = list(importances_sorted.values())
plt.barh(names[::-1], vals[::-1], color="#2c7fb8")
plt.xlabel("Importance (Gini-based)")
plt.title("Feature Importance — Decision Tree Model")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved feature_importance.png")

# ---------- 10. Confusion matrix chart ----------
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Test Set")
plt.colorbar()
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved confusion_matrix.png")

# ---------- 11. Save the trained model ----------
joblib.dump(model, "decision_tree_model.joblib")
print("Saved decision_tree_model.joblib")

# ---------- 12. Save readable tree rules (useful appendix content) ----------
tree_rules = export_text(model, feature_names=FEATURES)
with open("tree_rules.txt", "w") as f:
    f.write(tree_rules)
print("Saved tree_rules.txt")

print("\nDONE.")
