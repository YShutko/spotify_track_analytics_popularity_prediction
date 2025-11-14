"""
Train XGBoost Model with Artist Features

This script trains the model on the enriched dataset that includes artist features.

Expected improvement:
- Before (audio-only): R² = 0.16
- After (audio + artist): R² = 0.28-0.32 (~75-100% improvement)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎵 TRAINING WITH ARTIST FEATURES")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: Load Enriched Dataset
# ============================================================================
print("="*80)
print("📂 STEP 1: LOAD ENRICHED DATASET")
print("="*80)

df = pd.read_parquet('data/processed/cleaned_spotify_data_with_artists.parquet')
print(f"✓ Loaded {len(df):,} tracks")
print(f"✓ Columns ({len(df.columns)}): {list(df.columns)}")
print()

# ============================================================================
# STEP 2: Define Features
# ============================================================================
print("="*80)
print("🎯 STEP 2: DEFINE FEATURES")
print("="*80)

# Audio features (original 9)
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Artist features (new 4)
artist_features = [
    'artist_followers_log',  # Log-transformed followers
    'artist_popularity',      # Artist popularity score
    'artist_genre_count',     # Number of genres
    'artist_album_count'      # Discography size
]

# Combined feature set
all_features = audio_features + artist_features

print(f"Audio features ({len(audio_features)}): {audio_features}")
print(f"Artist features ({len(artist_features)}): {artist_features}")
print(f"Total features ({len(all_features)}): {all_features}")
print()

# Target variable
target = 'popularity'

# ============================================================================
# STEP 3: Prepare Data
# ============================================================================
print("="*80)
print("🔧 STEP 3: PREPARE DATA")
print("="*80)

# Extract features and target
X = df[all_features].copy()
y = df[target].copy()

# Check for missing values
missing = X.isnull().sum()
if missing.any():
    print("Missing values detected:")
    print(missing[missing > 0])
    print()

    # Fill missing values
    for col in X.columns:
        if X[col].isnull().any():
            if col in artist_features:
                X[col].fillna(X[col].median(), inplace=True)
                print(f"✓ Filled {col} with median")
            else:
                X[col].fillna(0, inplace=True)
                print(f"✓ Filled {col} with 0")

print()
print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Target range: {y.min():.1f} - {y.max():.1f} (mean: {y.mean():.1f})")
print()

# Feature statistics
print("Feature statistics:")
print(X.describe())
print()

# ============================================================================
# STEP 4: Train/Val/Test Split
# ============================================================================
print("="*80)
print("✂️  STEP 4: TRAIN/VAL/TEST SPLIT")
print("="*80)

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"✓ Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Val: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print()

# ============================================================================
# STEP 5: Train Model (Using Best Hyperparameters from Optuna)
# ============================================================================
print("="*80)
print("🤖 STEP 5: TRAIN MODEL")
print("="*80)

# Use best hyperparameters from previous Optuna tuning
best_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'max_depth': 9,
    'learning_rate': 0.0131,
    'n_estimators': 500,
    'min_child_weight': 7,
    'subsample': 0.689,
    'colsample_bytree': 0.798,
    'reg_alpha': 1.11e-07,
    'reg_lambda': 0.0241,
    'gamma': 4.35e-05
}

print("Training XGBoost with best hyperparameters...")
print(json.dumps({k: v for k, v in best_params.items() if k not in ['early_stopping_rounds']}, indent=2))
print()

model = XGBRegressor(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("✓ Model training complete")
print()

# ============================================================================
# STEP 6: Evaluate Model
# ============================================================================
print("="*80)
print("📊 STEP 6: EVALUATE MODEL")
print("="*80)

def evaluate_model(X, y, name=""):
    """Calculate metrics for a dataset"""
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Adjusted R²
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"{name} Metrics:")
    print(f"  R² = {r2:.4f}")
    print(f"  Adjusted R² = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    print()

    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'mae': mae
    }

train_metrics = evaluate_model(X_train, y_train, "Training")
val_metrics = evaluate_model(X_val, y_val, "Validation")
test_metrics = evaluate_model(X_test, y_test, "Test")

# ============================================================================
# STEP 7: Feature Importance Analysis
# ============================================================================
print("="*80)
print("🎯 STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))
print()

# Calculate importance by feature type
audio_importance = feature_importance[feature_importance['feature'].isin(audio_features)]['importance'].sum()
artist_importance = feature_importance[feature_importance['feature'].isin(artist_features)]['importance'].sum()

print(f"Importance by Feature Type:")
print(f"  Audio features: {audio_importance:.1%}")
print(f"  Artist features: {artist_importance:.1%}")
print()

# ============================================================================
# STEP 8: Compare with Audio-Only Baseline
# ============================================================================
print("="*80)
print("📈 STEP 8: COMPARE WITH BASELINE")
print("="*80)

baseline_r2 = 0.1619  # From audio-only model
improvement = test_metrics['r2'] - baseline_r2
improvement_pct = (improvement / baseline_r2) * 100

print(f"Performance Comparison:")
print(f"  Baseline (audio-only): R² = {baseline_r2:.4f}")
print(f"  Current (audio + artist): R² = {test_metrics['r2']:.4f}")
print(f"  Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")
print()

if test_metrics['r2'] > 0.25:
    print("🎉 SUCCESS! Artist features significantly improved model performance!")
elif test_metrics['r2'] > 0.20:
    print("✅ GOOD! Artist features provided moderate improvement")
else:
    print("⚠️  WARNING: Lower than expected improvement")

print()

# ============================================================================
# STEP 9: Save Model and Metadata
# ============================================================================
print("="*80)
print("💾 STEP 9: SAVE MODEL AND METADATA")
print("="*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save model
model_path = Path(f'outputs/models/xgb_model_with_artists_{timestamp}.joblib')
joblib.dump(model, model_path)
print(f"✓ Model saved: {model_path}")

# Save feature importance
fi_path = Path(f'outputs/models/feature_importance_with_artists_{timestamp}.csv')
feature_importance.to_csv(fi_path, index=False)
print(f"✓ Feature importance saved: {fi_path}")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'cleaned_spotify_data_with_artists',
    'n_samples': len(df),
    'n_features': len(all_features),
    'feature_names': all_features,
    'audio_features': audio_features,
    'artist_features': artist_features,
    'model_params': {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'},
    'metrics': {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    },
    'baseline_comparison': {
        'baseline_r2': baseline_r2,
        'current_r2': test_metrics['r2'],
        'improvement': improvement,
        'improvement_pct': improvement_pct
    },
    'feature_importance_summary': {
        'audio_features_total': float(audio_importance),
        'artist_features_total': float(artist_importance)
    },
    'data_shapes': {
        'train': list(X_train.shape),
        'val': list(X_val.shape),
        'test': list(X_test.shape)
    }
}

metadata_path = Path(f'outputs/metadata/xgb_metadata_with_artists_{timestamp}.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("📊 TRAINING SUMMARY")
print("="*80)

print(f"""
Model Performance:
  Test R²: {test_metrics['r2']:.4f}
  Test RMSE: {test_metrics['rmse']:.2f}
  Test MAE: {test_metrics['mae']:.2f}

Improvement Over Baseline:
  Baseline (audio-only): R² = {baseline_r2:.4f}
  Current (audio + artist): R² = {test_metrics['r2']:.4f}
  Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)

Feature Importance:
  Audio features: {audio_importance:.1%}
  Artist features: {artist_importance:.1%}

Top 3 Most Important Features:
  1. {feature_importance.iloc[0]['feature']}: {feature_importance.iloc[0]['importance']:.1%}
  2. {feature_importance.iloc[1]['feature']}: {feature_importance.iloc[1]['importance']:.1%}
  3. {feature_importance.iloc[2]['feature']}: {feature_importance.iloc[2]['importance']:.1%}

Files Saved:
  - {model_path}
  - {fi_path}
  - {metadata_path}
""")

print("="*80)
print(f"✅ TRAINING COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
