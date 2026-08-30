# Students' Academic Performance Prediction System
### Using the Decision Tree Algorithm

Final year project — Department of Computer Science, Faculty of Computing,
University of Port Harcourt.

**Student:** BALOGUN OLADIMEJI TEMITOPE  
**Matriculation Number:** U2022/5570093

## What this is
A web-based decision-support tool that predicts a student's likely academic
performance category (Low / Medium / High) from five academic inputs, using
a single, interpretable Decision Tree (CART, scikit-learn), trained on the
UCI Student Performance dataset (Cortez & Silva, 2008).

## Folder contents
- `app.py` — Flask application (routes, form handling, SQLite history)
- `train_model.py` — trains the Decision Tree on student-mat.csv and saves
  the model, evaluation metrics, and charts
- `student-mat.csv` — the real UCI dataset used for training/testing
- `decision_tree_model.joblib` — the trained model (already trained; you do
  not need to retrain to run the app)
- `model_results.json` — accuracy/precision/recall/F1, confusion matrix,
  feature importances (used by the app's About page, and by Chapter 4)
- `decision_tree_visualization.png`, `feature_importance.png`,
  `confusion_matrix.png` — figures for Chapter 4
- `tree_rules.txt` — the trained tree as readable if-then rules (appendix material)
- `templates/`, `static/style.css` — the web interface
- `test_app.py`, `test_results.txt` — the test suite and its (passing) output
- `screenshots/` — captured pages of the running app, for Chapter 4
- `predictions.db` — SQLite database (created automatically on first run)

## How to run it
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the app:
   ```
   python3 app.py
   ```
3. Open a browser to: http://127.0.0.1:5000

## How to retrain the model (optional — not required to run the app)
```
python3 train_model.py
```
This regenerates `decision_tree_model.joblib`, `model_results.json`, and the
three chart PNGs from `student-mat.csv`.

## How to run the tests
```
python3 -m pytest test_app.py -v
```
