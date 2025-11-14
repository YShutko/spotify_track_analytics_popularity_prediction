"""
Tune Random Forest with Optuna on Cleaned Dataset

This script uses Optuna to optimize Random Forest hyperparameters for
predicting song popularity. The goal is to see if tuned Random Forest can
approach or match XGBoost performance (R² = 0.1619).

Optimized hyperparameters:
- n_estimators: Number of trees
- max_depth: Maximum tree depth
- min_samples_split: Minimum samples to split node
- min_samples_leaf: Minimum samples in leaf
- max_features: Number of features per split
- bootstrap: Use bootstrap sampling

Baseline Random Forest (default): R² = 0.1315, severe overfitting
Target: XGBoost R² = 0.1619
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🌲 TUNING RANDOM FOREST WITH OPTUNA")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: Load Cleaned Dataset
# ============================================================================
print("="*80)
print("📂 STEP 1: LOAD CLEANED DATASET")
print("="*80)

df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')
print(f"✓ Loaded {len(df):,} tracks")
print()

# ============================================================================
# STEP 2: Prepare Features
# ============================================================================
print("="*80)
print("🎯 STEP 2: PREPARE FEATURES")
print("="*80)

# Use same 9 audio features as XGBoost model
feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

X = df[feature_cols].copy()
y = df['popularity'].copy()

# Remove NaN values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X, y = X[mask], y[mask]

print(f"✓ Features: {feature_cols}")
print(f"✓ Target: popularity")
print(f"✓ Samples: {len(X):,}")
print()

# ============================================================================
# STEP 3: Train/Val/Test Split
# ============================================================================
print("="*80)
print("✂️  STEP 3: TRAIN/VAL/TEST SPLIT")
print("="*80)

# Use same split as XGBoost: 70% train, 15% val, 15% test
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
# STEP 4: Define Optuna Objective
# ============================================================================
print("="*80)
print("🎯 STEP 4: OPTUNA HYPERPARAMETER TUNING")
print("="*80)

def objective(trial):
    """Optuna objective function for Random Forest"""

    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8, 1.0])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)

    return val_r2

# Run Optuna optimization
print("Starting Optuna optimization (50 trials)...")
print("This will take approximately 10-15 minutes...")
print()

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='random_forest_tuning'
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print()
print("✓ Optimization complete!")
print()

# ============================================================================
# STEP 5: Best Parameters
# ============================================================================
print("="*80)
print("🏆 STEP 5: BEST HYPERPARAMETERS")
print("="*80)

best_params = study.best_params
best_val_r2 = study.best_value

print("Best hyperparameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print()
print(f"Best validation R²: {best_val_r2:.4f}")
print()

# ============================================================================
# STEP 6: Train Final Model with Best Parameters
# ============================================================================
print("="*80)
print("🌲 STEP 6: TRAIN FINAL MODEL")
print("="*80)

final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training final Random Forest with best parameters...")
final_model.fit(X_train, y_train)
print("✓ Training complete")
print()

# ============================================================================
# STEP 7: Evaluate Final Model
# ============================================================================
print("="*80)
print("📊 STEP 7: EVALUATE FINAL MODEL")
print("="*80)

def evaluate_model(model, X, y, name=""):
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

train_metrics = evaluate_model(final_model, X_train, y_train, "Random Forest (tuned) - Training")
val_metrics = evaluate_model(final_model, X_val, y_val, "Random Forest (tuned) - Validation")
test_metrics = evaluate_model(final_model, X_test, y_test, "Random Forest (tuned) - Test")

# Calculate overfitting gap
overfitting_gap = train_metrics['r2'] - test_metrics['r2']
print(f"Overfitting gap: {overfitting_gap:.4f} (Train R² - Test R²)")
print()

# ============================================================================
# STEP 8: Compare with Baselines
# ============================================================================
print("="*80)
print("📈 STEP 8: COMPARISON WITH BASELINES")
print("="*80)

# Baseline metrics from previous training
lr_test_r2 = 0.0612
lr_test_rmse = 17.25
lr_test_mae = 14.21

rf_default_test_r2 = 0.1315
rf_default_test_rmse = 16.59
rf_default_test_mae = 13.25

xgb_test_r2 = 0.1619
xgb_test_rmse = 16.32
xgb_test_mae = 13.14

comparison_data = {
    'Model': [
        'Linear Regression',
        'Random Forest (default)',
        'Random Forest (tuned)',
        'XGBoost (tuned)'
    ],
    'Test R²': [
        lr_test_r2,
        rf_default_test_r2,
        test_metrics['r2'],
        xgb_test_r2
    ],
    'Test RMSE': [
        lr_test_rmse,
        rf_default_test_rmse,
        test_metrics['rmse'],
        xgb_test_rmse
    ],
    'Test MAE': [
        lr_test_mae,
        rf_default_test_mae,
        test_metrics['mae'],
        xgb_test_mae
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("Model Performance Comparison:")
print(comparison_df.to_string(index=False))
print()

# Determine best model
best_r2_idx = comparison_df['Test R²'].idxmax()
best_model_name = comparison_df.loc[best_r2_idx, 'Model']
print(f"🏆 Best Model: {best_model_name} (R² = {comparison_df.loc[best_r2_idx, 'Test R²']:.4f})")
print()

# Calculate improvements
rf_improvement = ((test_metrics['r2'] - rf_default_test_r2) / rf_default_test_r2) * 100
print(f"Improvement over default Random Forest: {rf_improvement:+.1f}%")

if test_metrics['r2'] > xgb_test_r2:
    xgb_improvement = ((test_metrics['r2'] - xgb_test_r2) / xgb_test_r2) * 100
    print(f"Improvement over XGBoost: {xgb_improvement:+.1f}%")
else:
    xgb_gap = ((xgb_test_r2 - test_metrics['r2']) / xgb_test_r2) * 100
    print(f"Gap to XGBoost: {xgb_gap:.1f}% (XGBoost still better)")
print()

# ============================================================================
# STEP 9: Feature Importance
# ============================================================================
print("="*80)
print("🎯 STEP 9: FEATURE IMPORTANCE")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 most important features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")
print()

# ============================================================================
# STEP 10: Save Results
# ============================================================================
print("="*80)
print("💾 STEP 10: SAVE RESULTS")
print("="*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save tuning results
results = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'cleaned_spotify_data_v2',
    'n_samples': len(df),
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'data_split': {
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0]
    },
    'optuna': {
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_params': best_params,
        'best_val_r2': best_val_r2
    },
    'final_model': {
        'hyperparameters': best_params,
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'overfitting_gap': overfitting_gap
    },
    'comparison': comparison_df.to_dict(orient='records'),
    'best_model': best_model_name,
    'feature_importance': feature_importance.to_dict(orient='records')
}

# Save to JSON
output_path = Path(f'outputs/metadata/rf_tuning_results_{timestamp}.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {output_path}")
print()

# Save Optuna study
study_path = Path(f'outputs/metadata/rf_optuna_study_{timestamp}.pkl')
import joblib
joblib.dump(study, study_path)
print(f"✓ Optuna study saved: {study_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("📊 RANDOM FOREST TUNING SUMMARY")
print("="*80)

print(f"""
Dataset: 78,310 cleaned tracks (deduplicated, zero-popularity removed)
Features: 9 audio features (same as XGBoost)
Optuna trials: {len(study.trials)}

RANDOM FOREST (default hyperparameters):
  Test R²:   {rf_default_test_r2:.4f}
  Test RMSE: {rf_default_test_rmse:.2f}
  Test MAE:  {rf_default_test_mae:.2f}
  Overfitting: Severe (Train R² = 0.88)

RANDOM FOREST (tuned with Optuna):
  Test R²:   {test_metrics['r2']:.4f}
  Test RMSE: {test_metrics['rmse']:.2f}
  Test MAE:  {test_metrics['mae']:.2f}
  Overfitting: {overfitting_gap:.4f} (Train R² - Test R²)

XGBOOST (tuned):
  Test R²:   {xgb_test_r2:.4f}
  Test RMSE: {xgb_test_rmse:.2f}
  Test MAE:  {xgb_test_mae:.2f}

BEST MODEL: {best_model_name}

Key Insights:
- Tuned Random Forest improves over default by {rf_improvement:+.1f}%
{'- Tuned Random Forest beats XGBoost by ' + f'{((test_metrics["r2"] - xgb_test_r2) / xgb_test_r2) * 100:+.1f}%' if test_metrics['r2'] > xgb_test_r2 else '- XGBoost still outperforms tuned Random Forest by ' + f'{((xgb_test_r2 - test_metrics["r2"]) / test_metrics["r2"]) * 100:.1f}%'}
- Overfitting reduced from 0.75 to {overfitting_gap:.4f}
- Hyperparameter tuning essential for Random Forest performance

Best hyperparameters:
  n_estimators: {best_params['n_estimators']}
  max_depth: {best_params['max_depth']}
  min_samples_split: {best_params['min_samples_split']}
  min_samples_leaf: {best_params['min_samples_leaf']}
  max_features: {best_params['max_features']}
  bootstrap: {best_params['bootstrap']}

All models still limited by audio-only features (R² ~0.16 ceiling).
Artist features expected to improve R² to 0.28-0.32.
""")

print("="*80)
print(f"✅ RANDOM FOREST TUNING COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
