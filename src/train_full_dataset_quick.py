"""
Quick Training on FULL Spotify Dataset (114K samples) - No Optuna

This is a faster version that uses good default hyperparameters.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_spotify_data.parquet"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METADATA_DIR = OUTPUTS_DIR / "metadata"

# Create directories
for dir_path in [MODELS_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🎵 QUICK TRAINING ON FULL SPOTIFY DATASET (114K SAMPLES)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("📂 Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded: {len(df):,} rows")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
target_col = 'popularity'

X = df[feature_cols].copy()
y = df[target_col].copy()

# Remove NaN
mask = ~(X.isnull().any(axis=1) | y.isnull())
X, y = X[mask], y[mask]

print(f"✅ Features: {len(feature_cols)}, Samples: {len(X):,}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n✂️  Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n🤖 Training XGBoost...")

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'objective': 'reg:squarederror',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

model = XGBRegressor(**params, early_stopping_rounds=20)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=10
)

print(f"\n✅ Training complete!")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n📊 Evaluating...")

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

def get_metrics(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    n, p = len(y_true), X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n{name}:")
    print(f"  R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.2f}, MAE = {mae:.2f}")
    print(f"  Predictions: [{y_pred.min():.2f}, {y_pred.max():.2f}], std={y_pred.std():.2f}")

    return {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae}

metrics_train = get_metrics(y_train, y_pred_train, "Train")
metrics_val = get_metrics(y_val, y_pred_val, "Validation")
metrics_test = get_metrics(y_test, y_pred_test, "Test")

# Health check
print(f"\n" + "="*80)
print("🔍 HEALTH CHECK")
print("="*80)

if metrics_test['r2'] > 0.3:
    print("✅ Model is healthy!")
    if metrics_test['r2'] > 0.7:
        print("🎉 Excellent performance!")
elif metrics_test['r2'] > 0:
    print("⚠️  Model is learning but performance is modest")
else:
    print("❌ Model has collapsed")

# ============================================================================
# SAVE
# ============================================================================
print("\n💾 Saving...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = MODELS_DIR / f"xgb_model_full_{timestamp}.joblib"
metadata_path = METADATA_DIR / f"xgb_metadata_full_{timestamp}.json"

joblib.dump(model, model_path)

metadata = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'full_spotify_114k',
    'n_samples': len(X),
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'model_params': params,
    'metrics': {
        'train_r2': metrics_train['r2'],
        'train_rmse': metrics_train['rmse'],
        'train_mae': metrics_train['mae'],
        'val_r2': metrics_val['r2'],
        'val_rmse': metrics_val['rmse'],
        'val_mae': metrics_val['mae'],
        'test_r2': metrics_test['r2'],
        'test_adjusted_r2': metrics_test['adj_r2'],
        'test_rmse': metrics_test['rmse'],
        'test_mae': metrics_test['mae'],
    },
    'data_shapes': {
        'train': list(X_train.shape),
        'val': list(X_val.shape),
        'test': list(X_test.shape)
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv(MODELS_DIR / f"feature_importance_full_{timestamp}.csv", index=False)

print(f"✅ Saved: {model_path.name}")
print(f"✅ Saved: {metadata_path.name}")

print(f"\n" + "="*80)
print("🎉 COMPLETE")
print("="*80)
print(f"Final Test R²: {metrics_test['r2']:.4f}")
print(f"Final Test RMSE: {metrics_test['rmse']:.2f}")
print(f"\nTop 3 Features:")
for _, row in feature_importance.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")
print("="*80)
