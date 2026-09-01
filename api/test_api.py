"""Tests for the prediction API (pytest + FastAPI TestClient)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


GOOD_PAYLOAD = {
    "prior_tier": 3,
    "registered_prev": 8,
    "passed_prev": 7,
    "carryover_prev": 1,
    "registered_curr": 8,
    "institution_name": "University of Lagos",
    "faculty": "Engineering",
    "qualification_type": "BSc",
    "programme_category": "Undergraduate",
    "current_academic_level": "300 level",
    "avg_grade_prev": "B",
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_good_student(client):
    resp = client.post("/predict", json=GOOD_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier"] in (0, 1, 2, 3, 4)
    assert data["cgpa_band"]
    assert data["class_name"]
    assert isinstance(data["at_risk"], bool)
    assert isinstance(data["at_risk_reason"], list)
    # Senior student with low carryover must NOT be flagged at risk.
    assert data["at_risk"] is False


def test_predict_known_at_risk_student(client):
    payload = dict(GOOD_PAYLOAD)
    payload.update(
        prior_tier=1, passed_prev=2, carryover_prev=5, avg_grade_prev="C"
    )
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["at_risk"] is True
    assert any("carryover" in r for r in data["at_risk_reason"])
    assert any("prior tier" in r for r in data["at_risk_reason"])


def test_predict_unknown_category_does_not_crash(client):
    payload = dict(GOOD_PAYLOAD)
    payload.update(
        institution_name="Never Heard University",
        faculty="Quantum Studies",
        avg_grade_prev="D",
    )
    resp = client.post("/predict", json=payload)
    # OneHotEncoder(handle_unknown="ignore") -> no crash, still a valid tier.
    assert resp.status_code == 200
    assert resp.json()["tier"] in (0, 1, 2, 3, 4)


@pytest.mark.parametrize("bad_field,bad_value", [
    ("prior_tier", 9),
    ("prior_tier", 0),
    ("carryover_prev", -1),
    ("registered_prev", 0),
    ("passed_prev", 25),
])
def test_predict_invalid_numeric_returns_422(client, bad_field, bad_value):
    payload = dict(GOOD_PAYLOAD)
    payload[bad_field] = bad_value
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_field_returns_422(client):
    payload = dict(GOOD_PAYLOAD)
    del payload["registered_curr"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_matches_trained_holdout_row(client):
    """Cross-check against a real corpus_test row: API must reproduce the
    model's own output for that exact student."""
    import pandas as pd

    from school_project.preprocess import build_clean_frames

    test = pd.read_csv("data/corpus_test.csv")
    X, y_tier, _ = build_clean_frames(test)
    row = test.iloc[0]
    from school_project.train import FEATURES

    Xrow = X.iloc[[0]][FEATURES]

    import joblib

    model = joblib.load("models/student_predictor.pkl")
    model_tier = int(model.predict(Xrow)[0])

    payload = {
        "prior_tier": int(row["prior_tier"]),
        "registered_prev": int(row["registered_prev"]),
        "passed_prev": int(row["passed_prev"]),
        "carryover_prev": int(row["carryover_prev"]),
        "registered_curr": int(row["registered_curr"]),
        "institution_name": str(row["institution_name"]),
        "faculty": str(row["faculty"]),
        "qualification_type": str(row["qualification_type"]),
        "programme_category": str(row["programme_category"]),
        "current_academic_level": str(row["current_academic_level"]),
        "avg_grade_prev": str(row["avg_grade_prev"]),
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["tier"] == model_tier