"""Load raw sensor data from the hydraulic systems dataset."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")


def load_ps2(path: Path | None = None) -> pd.DataFrame:
    """Load PS2 pressure sensor data (100 Hz, 6000 values per cycle).

    Args:
        path: Optional path override. Defaults to data/raw/PS2.txt.

    Returns:
        DataFrame of shape (n_cycles, 6000).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = path or RAW_DIR / "PS2.txt"
    logger.info(f"Loading PS2 from {file_path}")
    df = pd.read_csv(file_path, sep="\t", header=None)
    logger.info(f"PS2 loaded: {df.shape}")
    return df


def load_fs1(path: Path | None = None) -> pd.DataFrame:
    """Load FS1 flow rate sensor data (10 Hz, 600 values per cycle).

    Args:
        path: Optional path override. Defaults to data/raw/FS1.txt.

    Returns:
        DataFrame of shape (n_cycles, 600).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = path or RAW_DIR / "FS1.txt"
    logger.info(f"Loading FS1 from {file_path}")
    df = pd.read_csv(file_path, sep="\t", header=None)
    logger.info(f"FS1 loaded: {df.shape}")
    return df


def load_profile(path: Path | None = None) -> pd.DataFrame:
    """Load profile data containing target variables.

    Column index 3 = valve condition:
      100  → optimal   → class 1
      other → non-optimal → class 0

    Args:
        path: Optional path override. Defaults to data/raw/profile.txt.

    Returns:
        DataFrame of shape (n_cycles, n_targets).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = path or RAW_DIR / "profile.txt"
    logger.info(f"Loading profile from {file_path}")
    df = pd.read_csv(file_path, sep="\t", header=None)
    logger.info(f"Profile loaded: {df.shape}")
    return df


def load_all(
    ps2_path: Path | None = None,
    fs1_path: Path | None = None,
    profile_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three raw data files.

    Args:
        ps2_path: Optional path for PS2.txt.
        fs1_path: Optional path for FS1.txt.
        profile_path: Optional path for profile.txt.

    Returns:
        Tuple of (ps2, fs1, profile) DataFrames.
    """
    ps2 = load_ps2(ps2_path)
    fs1 = load_fs1(fs1_path)
    profile = load_profile(profile_path)

    n_cycles = len(profile)
    assert len(ps2) == n_cycles, f"PS2 cycle count mismatch: {len(ps2)} vs {n_cycles}"
    assert len(fs1) == n_cycles, f"FS1 cycle count mismatch: {len(fs1)} vs {n_cycles}"

    logger.info(f"All data loaded: {n_cycles} cycles")
    return ps2, fs1, profile


if __name__ == "__main__":
    ps2, fs1, profile = load_all()
    print(f"PS2:     {ps2.shape}")
    print(f"FS1:     {fs1.shape}")
    print(f"Profile: {profile.shape}")
    print(f"Valve condition distribution:\n{profile.iloc[:, 3].value_counts()}")
