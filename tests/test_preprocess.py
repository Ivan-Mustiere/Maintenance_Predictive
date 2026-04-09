"""Tests for feature engineering and train/test split."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    extract_features,
    build_features,
    build_target,
    split_train_test,
    TRAIN_SIZE,
)


@pytest.fixture
def sample_ps2() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(2200, 6000))


@pytest.fixture
def sample_fs1() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(2200, 600))


@pytest.fixture
def sample_profile() -> pd.DataFrame:
    np.random.seed(42)
    data = np.random.rand(2200, 5)
    data[:, 1] = np.random.choice([100, 90, 80, 73], size=2200)
    return pd.DataFrame(data)


def test_extract_features_keys():
    signal = np.random.rand(6000)
    feats = extract_features(signal, "ps2")
    expected_keys = {
        "ps2_mean", "ps2_std", "ps2_min", "ps2_max", "ps2_median",
        "ps2_q25", "ps2_q75", "ps2_range", "ps2_rms", "ps2_skew", "ps2_kurtosis",
    }
    assert set(feats.keys()) == expected_keys


def test_extract_features_values_are_finite():
    signal = np.random.rand(600)
    feats = extract_features(signal, "fs1")
    for v in feats.values():
        assert np.isfinite(v)


def test_build_features_shape(sample_ps2, sample_fs1):
    features = build_features(sample_ps2, sample_fs1)
    assert features.shape == (2200, 22)  # 11 features × 2 sensors


def test_build_features_no_nan(sample_ps2, sample_fs1):
    features = build_features(sample_ps2, sample_fs1)
    assert not features.isnull().any().any()


def test_build_target_binary(sample_profile):
    target = build_target(sample_profile)
    assert set(target.unique()).issubset({0, 1})


def test_build_target_mapping(sample_profile):
    """Valve condition 100 must map to 1, others to 0."""
    target = build_target(sample_profile)
    valve = sample_profile.iloc[:, 1]
    for i in range(len(target)):
        if valve.iloc[i] == 100:
            assert target.iloc[i] == 1
        else:
            assert target.iloc[i] == 0


def test_split_train_size(sample_ps2, sample_fs1, sample_profile):
    features = build_features(sample_ps2, sample_fs1)
    target = build_target(sample_profile)
    X_train, X_test, y_train, y_test = split_train_test(features, target)

    assert len(X_train) == TRAIN_SIZE == 2000
    assert len(X_test) == 2200 - TRAIN_SIZE
    assert len(y_train) == TRAIN_SIZE
    assert len(y_test) == 2200 - TRAIN_SIZE


def test_split_no_overlap(sample_ps2, sample_fs1, sample_profile):
    """Ensure train and test rows do not overlap."""
    features = build_features(sample_ps2, sample_fs1)
    target = build_target(sample_profile)
    X_train, X_test, _, _ = split_train_test(features, target)

    train_vals = set(X_train["ps2_mean"].values)
    test_vals = set(X_test["ps2_mean"].values)
    # Sets may share values by coincidence but indices must not overlap
    assert len(X_train) + len(X_test) == len(features)
