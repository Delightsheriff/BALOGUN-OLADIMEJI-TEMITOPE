"""
evaluate_new_data.py — tests the saved Decision Tree models on data they
were never trained on.

Usage:
    python3 evaluate_new_data.py                       # uses sample_data/student-por.csv
    python3 evaluate_new_data.py path/to/class.csv     # any CSV with the five inputs and G3

The default file is the Portuguese-language course file from the same UCI
Student Performance dataset (Cortez & Silva, 2008): 649 records from a
different subject, none of which were used in training. The models are
loaded as saved, not retrained.

Saves new_data_results.json and new_data_confusion_matrix.png.
"""
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictor import LABELS, read_student_csv, predict_class, score

path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/student-por.csv"
df = read_student_csv(path)
if "G3" not in df.columns:
    sys.exit("The file needs a G3 (final grade) column to measure accuracy.")

results = {"file": path, "records": int(len(df))}
for name, data in (("full_model", df), ("early_model", df.drop(columns=["G2"], errors="ignore"))):
    rows, skipped, mode = predict_class(data)
    s = score(rows)
    results[name] = {
        "mode": mode,
        "scored": s["n"],
        "skipped": len(skipped),
        "correct": s["correct"],
        "accuracy": s["accuracy"],
        "confusion_matrix": [[s["matrix"][a][p] for p in LABELS] for a in LABELS],
        "confusion_matrix_labels": LABELS,
    }
    print(f"{mode}: {s['correct']}/{s['n']} correct = {s['accuracy']}% accuracy")

with open("new_data_results.json", "w") as f:
    json.dump(results, f, indent=2)

cm = results["full_model"]["confusion_matrix"]
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
fig.colorbar(ax.images[0], ax=ax)
ax.set_title(f"Confusion Matrix — New Data ({results['records']} records)")
ax.set_xticks(range(3))
ax.set_xticklabels(LABELS)
ax.set_yticks(range(3))
ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
top = max(max(r) for r in cm)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                color="white" if cm[i][j] > top / 2 else "black")
plt.tight_layout()
plt.savefig("new_data_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved new_data_results.json and new_data_confusion_matrix.png")
