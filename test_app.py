"""
test_app.py — Unit and integration tests for the Academic Performance
Prediction System.

Run with:  python3 -m pytest test_app.py -v
"""
import os
import joblib
import pytest
import numpy as np

import app as flask_app_module

MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_tree_model.joblib")


# ---------------------------------------------------------------------------
# UNIT TESTS — the trained model in isolation
# ---------------------------------------------------------------------------

class TestModelUnit:
    @classmethod
    def setup_class(cls):
        cls.model = joblib.load(MODEL_PATH)

    def test_model_loads(self):
        assert self.model is not None

    def test_model_has_three_classes(self):
        assert set(self.model.classes_) == {"Low", "Medium", "High"}

    def test_predict_returns_valid_class(self):
        pred = self.model.predict([[3, 0, 2, 15, 16]])[0]
        assert pred in {"Low", "Medium", "High"}

    def test_high_grades_predict_high_or_medium(self):
        # A strong student: high study time, no failures, few absences, high grades
        pred = self.model.predict([[4, 0, 0, 18, 19]])[0]
        assert pred in {"High", "Medium"}

    def test_low_grades_predict_low(self):
        # A struggling student: low study time, past failures, high absences, low grades
        pred = self.model.predict([[1, 3, 25, 4, 3]])[0]
        assert pred == "Low"

    def test_predict_proba_sums_to_one(self):
        proba = self.model.predict_proba([[2, 0, 5, 12, 13]])[0]
        assert abs(sum(proba) - 1.0) < 1e-6

    def test_feature_importances_are_non_negative(self):
        assert all(imp >= 0 for imp in self.model.feature_importances_)

    def test_tree_depth_within_limit(self):
        # Guards against accidental overfitting if the model is ever retrained
        assert self.model.get_depth() <= 4


# ---------------------------------------------------------------------------
# INTEGRATION TESTS — the Flask routes end-to-end
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

    def test_predict_form_get(self, client):
        resp = client.get("/predict")
        assert resp.status_code == 200
        assert b"studytime" in resp.data

    def test_predict_form_post_valid_data(self, client):
        resp = client.post("/predict", data={
            "student_name": "Test Student",
            "studytime": "2",
            "failures": "0",
            "absences": "3",
            "g1": "14",
            "g2": "15",
        })
        assert resp.status_code == 200
        assert b"Prediction Result" in resp.data
        assert b"Test Student" in resp.data

    def test_predict_form_post_missing_field_shows_error(self, client):
        resp = client.post("/predict", data={
            "student_name": "Incomplete Student",
            "studytime": "2",
            # 'failures' deliberately omitted
            "absences": "3",
            "g1": "14",
            "g2": "15",
        })
        assert resp.status_code == 200
        assert b"valid numbers" in resp.data

    def test_history_page_loads(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200

    def test_about_page_loads_and_shows_metrics(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200
        assert b"Accuracy" in resp.data

    def test_prediction_is_persisted_to_history(self, client):
        client.post("/predict", data={
            "student_name": "Persistence Check",
            "studytime": "3", "failures": "0", "absences": "1",
            "g1": "16", "g2": "17",
        })
        resp = client.get("/history")
        assert b"Persistence Check" in resp.data
