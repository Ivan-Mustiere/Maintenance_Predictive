"""Feature engineering and train/test split for hydraulic system cycles."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from src.data.load_data import load_all
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
TRAIN_SIZE = 2000  # Fixed: first 2000 cycles are train, rest are test


def extract_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    """Extract statistical features from a 1D time-series signal.

    Args:
        signal: 1D array of sensor values for one cycle.
        prefix: Feature name prefix (e.g., 'ps2' or 'fs1').

    Returns:
        Dictionary of feature_name → value.
    """
    return {
        f"{prefix}_mean": float(np.mean(signal)),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_min": float(np.min(signal)),
        f"{prefix}_max": float(np.max(signal)),
        f"{prefix}_median": float(np.median(signal)),
        f"{prefix}_q25": float(np.percentile(signal, 25)),
        f"{prefix}_q75": float(np.percentile(signal, 75)),
        f"{prefix}_range": float(np.max(signal) - np.min(signal)),
        f"{prefix}_rms": float(np.sqrt(np.mean(signal**2))),
        f"{prefix}_skew": float(stats.skew(signal)),
        f"{prefix}_kurtosis": float(stats.kurtosis(signal)),
    }


def build_features(
    ps2: pd.DataFrame,
    fs1: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix from raw sensor DataFrames.

    Args:
        ps2: Raw PS2 DataFrame of shape (n_cycles, 6000).
        fs1: Raw FS1 DataFrame of shape (n_cycles, 600).

    Returns:
        Feature DataFrame of shape (n_cycles, n_features).
    """
    logger.info("Extracting features from PS2 and FS1...")
    rows: list[dict[str, float]] = []

    for i in range(len(ps2)):
        row: dict[str, float] = {}
        row.update(extract_features(ps2.iloc[i].values, "ps2"))
        row.update(extract_features(fs1.iloc[i].values, "fs1"))
        rows.append(row)

    features = pd.DataFrame(rows)
    logger.info(f"Feature matrix built: {features.shape}")
    return features


def build_target(profile: pd.DataFrame) -> pd.Series:
    """Build binary target from valve condition (column index 3).

    Valve condition 100 → 1 (optimal), anything else → 0 (non-optimal).

    Args:
        profile: Raw profile DataFrame.

    Returns:
        Binary target Series.
    """
    valve = profile.iloc[:, 3]
    target = (valve == 100).astype(int)
    logger.info(f"Target distribution:\n{target.value_counts().to_string()}")
    return target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets.

    Train: first 2000 cycles (index 0–1999).
    Test:  remaining cycles (index 2000+).

    Args:
        features: Full feature DataFrame.
        target: Full binary target Series.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train = features.iloc[:TRAIN_SIZE].reset_index(drop=True)
    X_test = features.iloc[TRAIN_SIZE:].reset_index(drop=True)
    y_train = target.iloc[:TRAIN_SIZE].reset_index(drop=True)
    y_test = target.iloc[TRAIN_SIZE:].reset_index(drop=True)

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_preprocessing() -> None:
    """Full preprocessing pipeline: load → features → split → save."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ps2, fs1, profile = load_all()
    features = build_features(ps2, fs1)
    target = build_target(profile)
    X_train, X_test, y_train, y_test = split_train_test(features, target)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    features.to_csv(PROCESSED_DIR / "features_all.csv", index=False)
    target.to_csv(PROCESSED_DIR / "target_all.csv", index=False)

    logger.info(f"Processed data saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    run_preprocessing()
