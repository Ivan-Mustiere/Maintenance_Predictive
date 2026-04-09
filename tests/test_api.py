"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import src.models.predict as predict_module


@pytest.fixture
def client():
    """Create a test client with mocked model loading."""
    with (
        patch.object(predict_module, "_load_model", return_value=None),
        patch.object(predict_module, "_load_features", return_value=None),
    ):
        from api.app import app
        with TestClient(app) as c:
            yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_format(client):
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data


def test_predict_valid_cycle(client):
    mock_result = {
        "cycle_id": 42,
        "prediction": "optimal",
        "probability": 0.93,
    }
    with patch("api.app.predict_cycle", return_value=mock_result):
        response = client.get("/predict?cycle_id=42")

    assert response.status_code == 200
    data = response.json()
    assert data["cycle_id"] == 42
    assert data["prediction"] in {"optimal", "non_optimal"}
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_response_fields(client):
    mock_result = {
        "cycle_id": 0,
        "prediction": "non_optimal",
        "probability": 0.75,
    }
    with patch("api.app.predict_cycle", return_value=mock_result):
        response = client.get("/predict?cycle_id=0")

    data = response.json()
    assert set(data.keys()) == {"cycle_id", "prediction", "probability"}


def test_predict_out_of_range_returns_404(client):
    with patch("api.app.predict_cycle", side_effect=IndexError("out of range")):
        response = client.get("/predict?cycle_id=99999")
    assert response.status_code == 404


def test_predict_missing_param_returns_422(client):
    response = client.get("/predict")
    assert response.status_code == 422


def test_predict_negative_cycle_returns_422(client):
    response = client.get("/predict?cycle_id=-1")
    assert response.status_code == 422


def test_model_info_returns_200(client):
    response = client.get("/model/info")
    assert response.status_code == 200
