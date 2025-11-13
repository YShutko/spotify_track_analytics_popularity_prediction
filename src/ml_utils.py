"""
ML Utility Functions for Spotify Track Analytics

This module provides utility functions for:
- Data validation
- Model evaluation metrics
- Metadata tracking
- Reproducibility helpers
"""

import subprocess
import platform
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


def validate_train_test_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """
    Validate that train and test sets have consistent features.

    Args:
        X_train: Training feature dataframe
        X_test: Test feature dataframe

    Raises:
        AssertionError: If features don't match
    """
    train_cols = list(X_train.columns)
    test_cols = list(X_test.columns)

    assert train_cols == test_cols, (
        f"❌ Feature mismatch between train/test sets\n"
        f"Train features: {set(train_cols)}\n"
        f"Test features: {set(test_cols)}\n"
        f"Missing in test: {set(train_cols) - set(test_cols)}\n"
        f"Extra in test: {set(test_cols) - set(train_cols)}"
    )

    print(f"✅ Features consistent across datasets ({len(train_cols)} features)")


def check_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """
    Check and report missing values in train and test sets.

    Args:
        X_train: Training feature dataframe
        X_test: Test feature dataframe
    """
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()

    print(f"Missing values in training data: {train_missing}")
    print(f"Missing values in test data: {test_missing}")

    if train_missing > 0 or test_missing > 0:
        print("⚠️  Warning: Missing values detected!")


def adjusted_r2(r2: float, n: int, k: int) -> float:
    """
    Calculate adjusted R² score.

    The adjusted R² accounts for the number of predictors in the model,
    providing a better metric for multivariate model comparison.

    Args:
        r2: R² score from model
        n: Number of samples
        k: Number of features

    Returns:
        Adjusted R² score

    Example:
        >>> adj_r2 = adjusted_r2(r2=0.85, n=1000, k=10)
        >>> print(f"Adjusted R²: {adj_r2:.4f}")
    """
    if n <= k + 1:
        raise ValueError(f"Sample size (n={n}) must be greater than features + 1 (k+1={k+1})")

    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def get_git_commit() -> str:
    """
    Get current git commit hash for reproducibility.

    Returns:
        Git commit hash (SHA-1) or None if not in git repo
    """
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return git_commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_environment_info() -> Dict[str, str]:
    """
    Collect environment information for reproducibility.

    Returns:
        Dictionary containing Python version, platform, and key library versions
    """
    try:
        import sklearn
        import xgboost
        import pandas as pd
        import numpy as np
        import shap

        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'scikit_learn_version': sklearn.__version__,
            'xgboost_version': xgboost.__version__,
            'shap_version': shap.__version__
        }

        return env_info
    except ImportError as e:
        print(f"⚠️  Warning: Could not import library: {e}")
        return {
            'python_version': sys.version,
            'platform': platform.platform()
        }


def create_model_metadata(
    model_params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_names: list,
    train_size: Tuple[int, int],
    test_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Create comprehensive model metadata for tracking.

    Args:
        model_params: Model hyperparameters
        metrics: Evaluation metrics (RMSE, MAE, R², etc.)
        feature_names: List of feature names used
        train_size: Training set shape (n_samples, n_features)
        test_size: Test set shape (n_samples, n_features)

    Returns:
        Metadata dictionary with all tracking information
    """
    from datetime import datetime

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'environment': get_environment_info(),
        'model_params': model_params,
        'metrics': metrics,
        'features': {
            'names': feature_names,
            'count': len(feature_names)
        },
        'data_shapes': {
            'train': train_size,
            'test': test_size
        }
    }

    return metadata


def save_model_with_metadata(
    model: Any,
    metadata: Dict[str, Any],
    model_path: str,
    metadata_path: str
) -> None:
    """
    Save model and metadata to disk.

    Args:
        model: Trained model object
        metadata: Model metadata dictionary
        model_path: Path to save model (.joblib)
        metadata_path: Path to save metadata (.json)
    """
    import joblib

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")

    # Print key info
    if metadata.get('git_commit'):
        print(f"📌 Git commit: {metadata['git_commit'][:8]}")
    print(f"📊 Metrics: {metadata.get('metrics', {})}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"✅ Loaded config from: {config_path}")
    return config


def log_data_split_info(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Log information about data splits.

    Args:
        X_train, X_val, X_test: Feature dataframes
        y_train, y_val, y_test: Target series
    """
    total_samples = len(X_train) + len(X_val) + len(X_test)

    print("\n" + "="*60)
    print("📊 DATA SPLIT SUMMARY")
    print("="*60)
    print(f"Training set:   {X_train.shape[0]:,} samples ({100*len(X_train)/total_samples:.1f}%)")
    print(f"Validation set: {X_val.shape[0]:,} samples ({100*len(X_val)/total_samples:.1f}%)")
    print(f"Test set:       {X_test.shape[0]:,} samples ({100*len(X_test)/total_samples:.1f}%)")
    print(f"Total:          {total_samples:,} samples")
    print(f"Features:       {X_train.shape[1]}")
    print("="*60 + "\n")

    # Check target distribution
    print("Target distribution (popularity):")
    print(f"  Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"  Val:   mean={y_val.mean():.2f}, std={y_val.std():.2f}")
    print(f"  Test:  mean={y_test.mean():.2f}, std={y_test.std():.2f}\n")
