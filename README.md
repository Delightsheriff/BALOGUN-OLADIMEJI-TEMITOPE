# Students' Academic Performance Prediction System
### Using the Decision Tree Algorithm

Final year project, Department of Computer Science, Faculty of Computing,
University of Port Harcourt.

**Student:** Balogun Oladimeji Temitope (U2022/5570093)

A Flask web application that predicts a student's performance category
(Low, Medium or High) from five academic inputs, using an interpretable CART
Decision Tree (scikit-learn) trained on the UCI Student Performance dataset
(Cortez & Silva, 2008). Every prediction shows the tree path that produced it.

## Results

| Model | Test set (79) | Training set (316) | New data (649) |
|---|---|---|---|
| Full-term (study time, failures, absences, G1, G2) | 82.3% | 88.0% | 85.2% |
| Early-warning (same, without G2) | 74.7% | 76.6% | 66.3% |

"New data" is `sample_data/student-por.csv`, the Portuguese-language course file
from the same UCI dataset. The models never saw it during training.

## Pages

- **Home**: what the system does, headline results.
- **New Prediction**: one student, full-term or early-warning mode.
- **Batch Upload**: a whole class from a CSV. Comma- or semicolon-separated files
  both work, including the raw UCI files. If the file has a `G3` column, the page
  also reports accuracy and a confusion matrix. "Run on 649 New Records" does this
  with the built-in new-data file.
- **History**: every prediction, searchable, filterable, exportable to CSV.
- **About the Model**: parameters, metrics, feature importance, project details,
  and a source-code download.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask routes, SQLite history, `/health`, source-code zip |
| `predictor.py` | Shared prediction logic: models, validation, CSV reading, decision path |
| `train_model.py` | Trains both trees on `student-mat.csv`; saves the models, `model_results.json`, charts and `tree_rules.txt` |
| `evaluate_new_data.py` | Tests the saved models on unseen data; saves `new_data_results.json` and a confusion matrix |
| `project.toml` | Student, supervisor, session and other details shown in the app |
| `sample_data/` | `student-por.csv` (new data) and `class_template.csv` |
| `test_app.py` | 27 unit and integration tests (`test_results.txt` holds the last run) |
| `render.yaml` | Render deployment (free plan, auto-deploy, health check) |
| `screenshots/` | Screens of the running app |

## Run locally

```
pip install -r requirements.txt
python3 app.py
```
Open http://127.0.0.1:5000

## Test, retrain, evaluate

```
python3 -m pytest test_app.py -v
python3 train_model.py          # optional: rebuilds the models and charts
python3 evaluate_new_data.py    # optional: any CSV path with the five inputs and G3
```

## Update the supervisor or session

Edit `project.toml` (on GitHub is fine) and commit. Render redeploys
automatically and the new details show on every page.

## Deploy on Render

1. In Render, choose **New > Blueprint** and select this repository.
   `render.yaml` sets the free plan, Python 3.12.10, the build and start commands
   and auto-deploy on every push to `main`.
2. Render checks `/health`, which returns `ok`.

**Keep it awake:** in UptimeRobot, add an HTTP(s) monitor for
`https://<your-app>.onrender.com/health` every 5 minutes. The free plan sleeps
after 15 minutes without traffic.

**Note:** the free plan's disk is reset on every restart or redeploy, so the
prediction history starts empty each time. The models and data are part of the
code and are not affected.
