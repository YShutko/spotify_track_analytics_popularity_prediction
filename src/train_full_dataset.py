"""
Train XGBoost Model on FULL Spotify Dataset (114K samples)

This script fixes the model collapse issue by:
1. Using the full 114,000 sample dataset
2. Removing problematic features (release_year doesn't exist)
3. Using proper feature engineering without unnecessary scaling
4. Creating aligned train/test splits
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import optuna

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_spotify_data.parquet"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"
METADATA_DIR = OUTPUTS_DIR / "metadata"

# Create directories
for dir_path in [OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🎵 TRAINING ON FULL SPOTIFY DATASET (114K SAMPLES)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: LOAD FULL DATASET
# ============================================================================
print("\n" + "="*80)
print("📂 STEP 1: LOAD FULL DATASET")
print("="*80)

df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded data: {len(df):,} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 2: FEATURE SELECTION
# ============================================================================
print("\n" + "="*80)
print("🔧 STEP 2: FEATURE SELECTION")
print("="*80)

# Use only core audio features that exist and are meaningful
# DO NOT use release_year - it doesn't exist in the dataset!
feature_cols = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

# Verify all features exist
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"⚠️  Missing features: {missing_features}")
    feature_cols = [f for f in feature_cols if f in df.columns]

target_col = 'popularity'

print(f"\n✓ Target: {target_col}")
print(f"✓ Features ({len(feature_cols)}): {feature_cols}")

# Extract features and target
X = df[feature_cols].copy()
y = df[target_col].copy()

# Remove any rows with NaN
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

print(f"\n✅ Clean dataset: {len(X):,} samples, {len(feature_cols)} features")

# Show feature distributions (should be natural ranges, NOT scaled)
print(f"\n✓ Feature ranges (should be natural, not scaled):")
for col in feature_cols:
    print(f"  {col}: [{X[col].min():.3f}, {X[col].max():.3f}] "
          f"(mean: {X[col].mean():.3f}, std: {X[col].std():.3f})")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("✂️  STEP 3: TRAIN/TEST SPLIT")
print("="*80)

# Split: 70% train, 15% validation, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/(1-0.15), random_state=RANDOM_STATE, shuffle=True
)

print(f"✓ Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"✓ Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Verify distributions are similar
print(f"\n✓ Target distribution check:")
print(f"  Train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
print(f"  Val mean: {y_val.mean():.2f}, std: {y_val.std():.2f}")
print(f"  Test mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")

# Save test/train data for later use
X_train.to_parquet(BASE_DIR / "data" / "processed" / "X_train_full.parquet")
X_test.to_parquet(BASE_DIR / "data" / "processed" / "X_test_full.parquet")
y_train.to_frame().to_parquet(BASE_DIR / "data" / "processed" / "y_train_full.parquet")
y_test.to_frame().to_parquet(BASE_DIR / "data" / "processed" / "y_test_full.parquet")
print(f"\n✅ Saved train/test splits to data/processed/")

# ============================================================================
# STEP 4: OPTUNA HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("🎯 STEP 4: OPTUNA HYPERPARAMETER TUNING")
print("="*80)

def objective(trial):
    """Optuna objective function"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,

        # Tunable hyperparameters
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    return rmse

print("Running Optuna optimization (50 trials)...")
study = optuna.create_study(direction='minimize', study_name='spotify_xgboost')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n✅ Best trial:")
print(f"  RMSE: {study.best_value:.4f}")
print(f"\n✅ Best hyperparameters:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  {key}: {value}")

# ============================================================================
# STEP 5: TRAIN FINAL MODEL
# ============================================================================
print("\n" + "="*80)
print("🤖 STEP 5: TRAIN FINAL MODEL")
print("="*80)

# Add fixed params
final_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    **best_params
}

# Train on full training set
model = XGBRegressor(**final_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    early_stopping_rounds=50,
    verbose=10
)

print(f"\n✅ Model trained with {model.best_iteration} iterations")

# ============================================================================
# STEP 6: EVALUATE
# ============================================================================
print("\n" + "="*80)
print("📊 STEP 6: COMPREHENSIVE EVALUATION")
print("="*80)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_true, y_pred, set_name):
    """Calculate comprehensive metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Adjusted R²
    n = len(y_true)
    p = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n{set_name} Set:")
    print(f"  R² = {r2:.4f}")
    print(f"  Adjusted R² = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    print(f"  Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"  Prediction std: {y_pred.std():.2f}")

    return {
        'r2': r2,
        'adjusted_r2': adj_r2,
        'rmse': rmse,
        'mae': mae
    }

metrics = {}
metrics['train'] = calculate_metrics(y_train, y_pred_train, "Training")
metrics['val'] = calculate_metrics(y_val, y_pred_val, "Validation")
metrics['test'] = calculate_metrics(y_test, y_pred_test, "Test")

# Check for collapse
print(f"\n" + "="*80)
print("🔍 MODEL HEALTH CHECK")
print("="*80)

collapse_indicators = []

if metrics['test']['r2'] < 0:
    collapse_indicators.append("❌ Negative R² (model collapsed)")
elif metrics['test']['r2'] < 0.1:
    collapse_indicators.append("⚠️  Very low R² (<0.1)")

if y_pred_test.std() < 5:
    collapse_indicators.append(f"❌ Low prediction variance ({y_pred_test.std():.2f})")

pred_range = y_pred_test.max() - y_pred_test.min()
if pred_range < 20:
    collapse_indicators.append(f"❌ Narrow prediction range ({pred_range:.2f})")

if not collapse_indicators:
    print("✅ Model is healthy! No collapse detected.")
else:
    print("⚠️  Potential issues:")
    for indicator in collapse_indicators:
        print(f"  {indicator}")

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================
print("\n" + "="*80)
print("💾 STEP 7: SAVE MODEL & METADATA")
print("="*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = MODELS_DIR / f"xgb_model_full_{timestamp}.joblib"
metadata_path = METADATA_DIR / f"xgb_metadata_full_{timestamp}.json"

# Save model
joblib.dump(model, model_path)
print(f"✅ Model saved: {model_path}")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'full_spotify_114k',
    'n_samples': len(X),
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'model_params': final_params,
    'metrics': {
        'train_r2': metrics['train']['r2'],
        'train_rmse': metrics['train']['rmse'],
        'train_mae': metrics['train']['mae'],
        'val_r2': metrics['val']['r2'],
        'val_rmse': metrics['val']['rmse'],
        'val_mae': metrics['val']['mae'],
        'test_r2': metrics['test']['r2'],
        'test_adjusted_r2': metrics['test']['adjusted_r2'],
        'test_rmse': metrics['test']['rmse'],
        'test_mae': metrics['test']['mae'],
    },
    'data_shapes': {
        'train': list(X_train.shape),
        'val': list(X_val.shape),
        'test': list(X_test.shape)
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance_path = MODELS_DIR / f"feature_importance_full_{timestamp}.csv"
feature_importance.to_csv(feature_importance_path, index=False)
print(f"✅ Feature importance saved: {feature_importance_path}")

print(f"\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
print(f"\nFinal Test Performance:")
print(f"  R² = {metrics['test']['r2']:.4f}")
print(f"  RMSE = {metrics['test']['rmse']:.2f}")
print(f"  MAE = {metrics['test']['mae']:.2f}")
print(f"\nTop 5 Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
