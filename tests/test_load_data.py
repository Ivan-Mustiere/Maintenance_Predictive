"""Tests for data loading functions."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import io

from src.data.load_data import load_ps2, load_fs1, load_profile, load_all


def _make_txt(n_rows: int, n_cols: int) -> io.StringIO:
    """Helper to create a tab-separated in-memory file."""
    data = np.random.rand(n_rows, n_cols)
    lines = ["\t".join(map(str, row)) for row in data]
    return io.StringIO("\n".join(lines))


@pytest.fixture
def tmp_raw_files(tmp_path: Path) -> dict[str, Path]:
    """Create temporary raw data files for testing."""
    n_cycles = 10

    ps2_data = np.random.rand(n_cycles, 6000)
    fs1_data = np.random.rand(n_cycles, 600)
    # profile: 6 columns, valve condition at index 3
    profile_data = np.random.rand(n_cycles, 6)
    profile_data[:, 3] = np.random.choice([100, 90, 80, 73], size=n_cycles)

    ps2_path = tmp_path / "PS2.txt"
    fs1_path = tmp_path / "FS1.txt"
    profile_path = tmp_path / "profile.txt"

    pd.DataFrame(ps2_data).to_csv(ps2_path, sep="\t", header=False, index=False)
    pd.DataFrame(fs1_data).to_csv(fs1_path, sep="\t", header=False, index=False)
    pd.DataFrame(profile_data).to_csv(profile_path, sep="\t", header=False, index=False)

    return {"ps2": ps2_path, "fs1": fs1_path, "profile": profile_path}


def test_load_ps2_shape(tmp_raw_files):
    df = load_ps2(tmp_raw_files["ps2"])
    assert df.shape == (10, 6000)


def test_load_fs1_shape(tmp_raw_files):
    df = load_fs1(tmp_raw_files["fs1"])
    assert df.shape == (10, 600)


def test_load_profile_shape(tmp_raw_files):
    df = load_profile(tmp_raw_files["profile"])
    assert df.shape[0] == 10
    assert df.shape[1] >= 4


def test_no_nan_ps2(tmp_raw_files):
    df = load_ps2(tmp_raw_files["ps2"])
    assert not df.isnull().any().any()


def test_no_nan_fs1(tmp_raw_files):
    df = load_fs1(tmp_raw_files["fs1"])
    assert not df.isnull().any().any()


def test_no_nan_profile(tmp_raw_files):
    df = load_profile(tmp_raw_files["profile"])
    assert not df.isnull().any().any()


def test_load_all_consistent_cycle_count(tmp_raw_files):
    ps2, fs1, profile = load_all(
        tmp_raw_files["ps2"], tmp_raw_files["fs1"], tmp_raw_files["profile"]
    )
    assert len(ps2) == len(fs1) == len(profile) == 10


def test_load_ps2_file_not_found():
    with pytest.raises(Exception):
        load_ps2(Path("/nonexistent/PS2.txt"))
