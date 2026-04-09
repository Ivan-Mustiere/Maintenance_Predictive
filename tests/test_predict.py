"""Tests for the prediction module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import src.models.predict as predict_module


@pytest.fixture
def mock_model() -> Pipeline:
    """Train a tiny real model for testing."""
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    X = pd.DataFrame(np.random.rand(20, 22), columns=[f"f{i}" for i in range(22)])
    y = np.array([0, 1] * 10)
    clf.fit(X, y)
    return clf


@pytest.fixture
def mock_features() -> pd.DataFrame:
    np.random.seed(0)
    cols = [f"f{i}" for i in range(22)]
    return pd.DataFrame(np.random.rand(100, 22), columns=cols)


def test_predict_cycle_returns_valid_keys(mock_model, mock_features):
    with (
        patch.object(predict_module, "_load_model", return_value=mock_model),
        patch.object(predict_module, "_load_features", return_value=mock_features),
    ):
        result = predict_module.predict_cycle(0)

    assert "cycle_id" in result
    assert "prediction" in result
    assert "probability" in result


def test_predict_cycle_label_is_binary(mock_model, mock_features):
    with (
        patch.object(predict_module, "_load_model", return_value=mock_model),
        patch.object(predict_module, "_load_features", return_value=mock_features),
    ):
        result = predict_module.predict_cycle(5)

    assert result["prediction"] in {"optimal", "non_optimal"}


def test_predict_cycle_probability_in_range(mock_model, mock_features):
    with (
        patch.object(predict_module, "_load_model", return_value=mock_model),
        patch.object(predict_module, "_load_features", return_value=mock_features),
    ):
        result = predict_module.predict_cycle(10)

    assert 0.0 <= result["probability"] <= 1.0


def test_predict_cycle_correct_cycle_id(mock_model, mock_features):
    with (
        patch.object(predict_module, "_load_model", return_value=mock_model),
        patch.object(predict_module, "_load_features", return_value=mock_features),
    ):
        result = predict_module.predict_cycle(42)

    assert result["cycle_id"] == 42


def test_predict_cycle_out_of_range(mock_model, mock_features):
    with (
        patch.object(predict_module, "_load_model", return_value=mock_model),
        patch.object(predict_module, "_load_features", return_value=mock_features),
    ):
        with pytest.raises(IndexError):
            predict_module.predict_cycle(9999)
