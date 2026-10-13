"""
test_app.py — Unit and integration tests for the Academic Performance
Prediction System.

Run with:  python3 -m pytest test_app.py -v
"""
import os
import tempfile
import zipfile
import joblib
import pytest
import io
import pandas as pd
import numpy as np

# Use a throwaway database so tests never write into the real history.
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_predictions.db")

import app as flask_app_module
import predictor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_tree_model.joblib")
EARLY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "early_warning_model.joblib")


# ---------------------------------------------------------------------------
# UNIT TESTS — Trained Models in Isolation
# ---------------------------------------------------------------------------

class TestFullModelUnit:
    @classmethod
    def setup_class(cls):
        cls.model = joblib.load(MODEL_PATH)
        cls.features = ["studytime", "failures", "absences", "G1", "G2"]

    def test_model_loads(self):
        assert self.model is not None

    def test_model_has_three_classes(self):
        assert set(self.model.classes_) == {"Low", "Medium", "High"}

    def test_predict_returns_valid_class(self):
        df = pd.DataFrame([[3, 0, 2, 15, 16]], columns=self.features)
        pred = self.model.predict(df)[0]
        assert pred in {"Low", "Medium", "High"}

    def test_high_grades_predict_high_or_medium(self):
        df = pd.DataFrame([[4, 0, 0, 18, 19]], columns=self.features)
        pred = self.model.predict(df)[0]
        assert pred in {"High", "Medium"}

    def test_low_grades_predict_low(self):
        df = pd.DataFrame([[1, 3, 25, 4, 3]], columns=self.features)
        pred = self.model.predict(df)[0]
        assert pred == "Low"

    def test_predict_proba_sums_to_one(self):
        df = pd.DataFrame([[2, 0, 5, 12, 13]], columns=self.features)
        proba = self.model.predict_proba(df)[0]
        assert abs(sum(proba) - 1.0) < 1e-6

    def test_tree_depth_within_limit(self):
        assert self.model.get_depth() <= 4


class TestEarlyModelUnit:
    @classmethod
    def setup_class(cls):
        cls.model = joblib.load(EARLY_MODEL_PATH)
        cls.features = ["studytime", "failures", "absences", "G1"]

    def test_early_model_loads(self):
        assert self.model is not None

    def test_early_model_predicts_valid_class(self):
        df = pd.DataFrame([[3, 0, 1, 16]], columns=self.features)
        pred = self.model.predict(df)[0]
        assert pred in {"Low", "Medium", "High"}

    def test_early_model_detects_low_performance(self):
        df = pd.DataFrame([[1, 3, 20, 5]], columns=self.features)
        pred = self.model.predict(df)[0]
        assert pred == "Low"


class TestExplainabilityUnit:
    def test_explain_decision_path_returns_rules(self):
        model = joblib.load(MODEL_PATH)
        features = ["studytime", "failures", "absences", "G1", "G2"]
        df = pd.DataFrame([[2, 0, 4, 13, 12]], columns=features)
        rules, top_factor, advice = predictor.explain_decision_path(model, features, df)
        assert len(rules) > 0
        assert top_factor is not None
        assert isinstance(advice, list)


# ---------------------------------------------------------------------------
# INTEGRATION TESTS — Flask Endpoints & Flows
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as client:
        yield client


class TestRoutesIntegration:
    def test_home_page_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Academic Performance" in resp.data

    def test_predict_form_get_full_and_early(self, client):
        resp_full = client.get("/predict?mode=full")
        assert resp_full.status_code == 200
        assert b"Second Period Grade" in resp_full.data

        resp_early = client.get("/predict?mode=early")
        assert resp_early.status_code == 200
        assert b"Early-Warning Mode" in resp_early.data

    def test_predict_post_full_mode(self, client):
        resp = client.post("/predict", data={
            "mode": "full",
            "student_name": "Test Full Student",
            "studytime": "3",
            "failures": "0",
            "absences": "2",
            "g1": "15",
            "g2": "16",
        })
        assert resp.status_code == 200
        assert b"Prediction Result" in resp.data
        assert b"Test Full Student" in resp.data
        assert b"Decision-Tree Path" in resp.data

    def test_predict_post_early_warning_mode(self, client):
        resp = client.post("/predict", data={
            "mode": "early",
            "student_name": "Test Early Student",
            "studytime": "2",
            "failures": "1",
            "absences": "10",
            "g1": "8",
        })
        assert resp.status_code == 200
        assert b"Early-Warning" in resp.data
        assert b"Test Early Student" in resp.data

    def test_predict_post_invalid_grades_error(self, client):
        resp = client.post("/predict", data={
            "mode": "full",
            "student_name": "Invalid Student",
            "studytime": "2",
            "failures": "0",
            "absences": "2",
            "g1": "25",  # Invalid > 20
            "g2": "15",
        })
        assert resp.status_code == 200
        assert b"between 0 and 20" in resp.data

    def test_batch_predict_csv_upload(self, client):
        csv_data = (
            "student_name,studytime,failures,absences,g1,g2\n"
            "Student Alpha,3,0,1,16,17\n"
            "Student Beta,1,2,15,6,7\n"
        )
        data = {
            "file": (io.BytesIO(csv_data.encode("utf-8")), "class_roster.csv")
        }
        resp = client.post("/batch-predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert b"Student Alpha" in resp.data
        assert b"Student Beta" in resp.data
        assert b"Batch Classification Results" in resp.data

    def test_history_search_and_filter(self, client):
        client.post("/predict", data={
            "mode": "full",
            "student_name": "Searchable Student",
            "studytime": "2", "failures": "0", "absences": "2",
            "g1": "14", "g2": "14"
        })
        resp = client.get("/history?q=Searchable")
        assert resp.status_code == 200
        assert b"Searchable Student" in resp.data

    def test_history_export_csv(self, client):
        resp = client.get("/history/export")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "text/csv; charset=utf-8"
        assert b"Student Name" in resp.data


    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.data == b"ok"

    def test_project_details_come_from_config(self, client):
        resp = client.get("/about")
        assert flask_app_module.project["matric_number"].encode() in resp.data
        assert flask_app_module.project["supervisor"].encode() in resp.data

    def test_predict_rejects_non_numeric_input(self, client):
        resp = client.post("/predict", data={
            "mode": "full", "studytime": "2", "failures": "0",
            "absences": "abc", "g1": "12", "g2": "12",
        })
        assert b"must be a whole number" in resp.data

    def test_batch_reads_semicolon_uci_file_and_scores_it(self, client):
        csv_data = (
            '"school";"studytime";"failures";"absences";"G1";"G2";"G3"\n'
            '"GP";2;0;4;15;16;16\n'
            '"GP";1;3;20;4;3;0\n'
        )
        data = {"file": (io.BytesIO(csv_data.encode()), "uci_extract.csv")}
        resp = client.post("/batch-predict", data=data, content_type="multipart/form-data")
        assert b"Accuracy against the actual final grades" in resp.data
        assert b"Full Term model" in resp.data

    def test_batch_skips_invalid_rows(self, client):
        csv_data = "studytime,failures,absences,g1\n2,0,3,12\n9,0,3,12\n"
        data = {"file": (io.BytesIO(csv_data.encode()), "early.csv")}
        resp = client.post("/batch-predict", data=data, content_type="multipart/form-data")
        assert b"Early Warning model" in resp.data
        assert b"1 row(s) skipped" in resp.data

    def test_batch_sample_new_data(self, client):
        resp = client.post("/batch-predict", data={"sample": "1"})
        assert b"649" in resp.data
        assert b"Accuracy against the actual final grades" in resp.data

    def test_source_code_download(self, client):
        resp = client.get("/download/source")
        assert resp.status_code == 200
        names = zipfile.ZipFile(io.BytesIO(resp.data)).namelist()
        assert "academic-performance-system/app.py" in names
        assert not any(n.endswith(".db") for n in names)


class TestNewDataUnit:
    def test_full_model_accuracy_on_unseen_records(self):
        df = predictor.read_student_csv("sample_data/student-por.csv")
        rows, skipped, mode = predictor.predict_class(df)
        result = predictor.score(rows)
        assert result["n"] == 649 and not skipped
        assert result["accuracy"] >= 80
