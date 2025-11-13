"""
Improved ML Pipeline for Spotify Track Popularity Prediction with MLflow Tracking

This script implements all Phase 1 and Phase 2 improvements from the specification
with full MLflow experiment tracking integration (Phase 3):
- Data validation and separate validation set
- XGBoost model with JSON config
- Learning curves and diagnostic plots
- SHAP values for interpretability
- Comprehensive evaluation metrics
- Git commit tracking and environment metadata
- MLflow experiment tracking
"""

import os
import sys
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_utils import (
    validate_train_test_features,
    check_missing_values,
    adjusted_r2,
    create_model_metadata,
    save_model_with_metadata,
    load_config,
    log_data_split_info
)
from src.mlflow_tracker import MLflowTracker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "xgboost_params.json")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_music_data.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
METADATA_DIR = os.path.join(OUTPUTS_DIR, "metadata")

# Create output directories
for directory in [PLOTS_DIR, MODELS_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("🎵 IMPROVED ML PIPELINE WITH MLFLOW - SPOTIFY TRACK POPULARITY PREDICTION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# INITIALIZE MLFLOW TRACKING
# ============================================================================

print("\n" + "="*80)
print("🔬 INITIALIZE MLFLOW TRACKING")
print("="*80)

tracker = MLflowTracker(
    experiment_name="spotify_popularity_prediction",
    tracking_uri="sqlite:///mlruns/mlflow.db"
)

run = tracker.start_run(
    run_name=f"improved_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags={
        'dataset': 'real_data',
        'model_type': 'xgboost',
        'pipeline_version': 'improved_v1.0_mlflow'
    }
)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 1: LOAD DATA")
print("="*80)

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Cleaned data not found at {DATA_PATH}")
    print("Please run: make prepare-data")
    tracker.end_run(status="FAILED")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Display basic info
print(f"\nColumns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"\nColumns: {list(df.columns)}")


# ============================================================================
# STEP 2: FEATURE SELECTION & PREPARATION
# ============================================================================

print("\n" + "="*80)
print("🔧 STEP 2: FEATURE SELECTION & PREPARATION")
print("="*80)

# Define target and features
target_col = 'popularity'

# Select audio features (standardized in EDA)
audio_features = [
    'danceability', 'energy', 'loudness', 'acousticness',
    'tempo', 'valence', 'instrumentalness'
]

# Add additional numerical features if available
additional_features = ['duration_min', 'release_year', 'speechiness', 'liveness']
audio_features += [f for f in additional_features if f in df.columns]

# Remove any features with missing values or infinite values
audio_features = [f for f in audio_features if f in df.columns]

print(f"Target: {target_col}")
print(f"Features ({len(audio_features)}): {audio_features}")

# Prepare X and y
X = df[audio_features].copy()
y = df[target_col].copy()

# Remove any rows with NaN or infinite values
mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1) | y.isnull() | np.isinf(y))
X = X[mask]
y = y[mask]

print(f"\n✅ Clean dataset: {len(X):,} samples, {len(audio_features)} features")

# Log dataset information to MLflow
tracker.log_dict({
    'n_samples': len(X),
    'n_features': len(audio_features),
    'feature_names': audio_features,
    'target': target_col
}, 'dataset_info.json')


# ============================================================================
# STEP 3: DATA SPLITTING (WITH SEPARATE VALIDATION SET)
# ============================================================================

print("\n" + "="*80)
print("✂️  STEP 3: DATA SPLITTING (Phase 1 Improvement)")
print("="*80)

# Split into train+val (80%) and test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Split train_full into train (80%) and validation (20%)
# This gives us: 64% train, 16% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
)

# Log split information
log_data_split_info(X_train, X_val, X_test, y_train, y_val, y_test)

# Log split info to MLflow
tracker.log_params({
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_test),
    'train_ratio': 0.64,
    'val_ratio': 0.16,
    'test_ratio': 0.20
})

# PHASE 1: Data validation
print("🔍 PHASE 1: Data Validation")
validate_train_test_features(X_train, X_test)
validate_train_test_features(X_train, X_val)
check_missing_values(X_train, X_test)


# ============================================================================
# STEP 4: LOAD MODEL CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  STEP 4: LOAD MODEL CONFIGURATION (Phase 2 Improvement)")
print("="*80)

# PHASE 2: Load from JSON config
params = load_config(CONFIG_PATH)
print(f"\nModel Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Log parameters to MLflow
tracker.log_params(params)


# ============================================================================
# STEP 5: TRAIN MODEL WITH LEARNING CURVES
# ============================================================================

print("\n" + "="*80)
print("🤖 STEP 5: TRAIN XGBOOST MODEL")
print("="*80)

# Initialize model
model = XGBRegressor(**params)

# Train with evaluation set to capture learning curves
print("Training model with early stopping...")
eval_set = [(X_train, y_train), (X_val, y_val)]

model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

print(f"✅ Model trained successfully!")
print(f"Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

# Log model training info
if hasattr(model, 'best_iteration'):
    tracker.log_metrics({'best_iteration': model.best_iteration})


# PHASE 2: Learning Curves
print("\n📈 PHASE 2: Generating Learning Curves...")

results = model.evals_result()

plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['rmse'], label='Train RMSE', linewidth=2)
plt.plot(results['validation_1']['rmse'], label='Validation RMSE', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
learning_curve_path = os.path.join(PLOTS_DIR, 'xgboost_learning_curve.png')
plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
print(f"✅ Learning curve saved: {learning_curve_path}")

# Log plot to MLflow
tracker.log_figure(plt.gcf(), 'learning_curve.png')
plt.close()


# ============================================================================
# STEP 6: PREDICTIONS & EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 6: PREDICTIONS & EVALUATION")
print("="*80)

# Make predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate standard metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n📏 Standard Metrics:")
print(f"{'Metric':<15} {'Train':<12} {'Validation':<12} {'Test':<12}")
print("-" * 55)
print(f"{'RMSE':<15} {train_rmse:<12.4f} {val_rmse:<12.4f} {test_rmse:<12.4f}")
print(f"{'MAE':<15} {train_mae:<12.4f} {val_mae:<12.4f} {test_mae:<12.4f}")
print(f"{'R²':<15} {train_r2:<12.4f} {val_r2:<12.4f} {test_r2:<12.4f}")

# PHASE 2: Adjusted R²
print("\n📏 PHASE 2: Adjusted R² Metric:")
test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test.shape[1])
val_adj_r2 = adjusted_r2(val_r2, len(y_val), X_val.shape[1])

print(f"Test Adjusted R²: {test_adj_r2:.4f} (vs R²: {test_r2:.4f})")
print(f"Val Adjusted R²:  {val_adj_r2:.4f} (vs R²: {val_r2:.4f})")

# Store metrics for metadata and MLflow
metrics = {
    'train_rmse': float(train_rmse),
    'val_rmse': float(val_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'val_mae': float(val_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'val_r2': float(val_r2),
    'test_r2': float(test_r2),
    'test_adjusted_r2': float(test_adj_r2),
    'val_adjusted_r2': float(val_adj_r2)
}

# Log all metrics to MLflow
tracker.log_metrics(metrics)


# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 7: VISUALIZATIONS")
print("="*80)

# 7.1: Actual vs Predicted (Test Set)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
actual_pred_path = os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')
plt.savefig(actual_pred_path, dpi=300, bbox_inches='tight')
print(f"✅ Actual vs Predicted plot saved")
tracker.log_figure(plt.gcf(), 'actual_vs_predicted.png')
plt.close()

# PHASE 2: 7.2 Scatter Density Plot
print("\n📊 PHASE 2: Generating Scatter Density Plot...")
plt.figure(figsize=(10, 6))
sns.kdeplot(x=y_test, y=y_test_pred, cmap="Blues", fill=True, thresh=0.05, levels=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Prediction Density Plot (Test Set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
density_path = os.path.join(PLOTS_DIR, 'prediction_density.png')
plt.savefig(density_path, dpi=300, bbox_inches='tight')
print(f"✅ Density plot saved")
tracker.log_figure(plt.gcf(), 'prediction_density.png')
plt.close()

# 7.3: Residuals Plot
residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.3, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Popularity')
plt.ylabel('Residuals')
plt.title('Residual Plot (Test Set)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
residuals_path = os.path.join(PLOTS_DIR, 'residuals_plot.png')
plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
print(f"✅ Residuals plot saved")
tracker.log_figure(plt.gcf(), 'residuals_plot.png')
plt.close()

# PHASE 2: 7.4 QQ Plot for Residuals
print("\n📊 PHASE 2: Generating QQ Plot...")
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.grid(True, alpha=0.3)
plt.tight_layout()
qq_path = os.path.join(PLOTS_DIR, 'qq_plot_residuals.png')
plt.savefig(qq_path, dpi=300, bbox_inches='tight')
print(f"✅ QQ plot saved")
tracker.log_figure(plt.gcf(), 'qq_plot_residuals.png')
plt.close()

# PHASE 2: 7.5 Correlation Heatmap (Features + Target)
print("\n📊 PHASE 2: Generating Correlation Heatmap...")
analysis_df = X_train.copy()
analysis_df['popularity'] = y_train.values

plt.figure(figsize=(12, 10))
correlation_matrix = analysis_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, mask=mask,
            cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (with Target)')
plt.tight_layout()
corr_path = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
print(f"✅ Correlation heatmap saved")
tracker.log_figure(plt.gcf(), 'correlation_heatmap.png')
plt.close()


# ============================================================================
# STEP 8: FEATURE IMPORTANCE (STANDARD)
# ============================================================================

print("\n" + "="*80)
print("🔍 STEP 8: FEATURE IMPORTANCE (Standard)")
print("="*80)

# Standard feature importance
feature_importance = pd.DataFrame({
    'feature': audio_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features (Standard Importance):")
print(feature_importance.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('XGBoost Feature Importance (Gain)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
feature_imp_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
plt.savefig(feature_imp_path, dpi=300, bbox_inches='tight')
print(f"✅ Feature importance plot saved")
tracker.log_figure(plt.gcf(), 'feature_importance.png')
plt.close()


# ============================================================================
# STEP 9: SHAP VALUES (CRITICAL - PHASE 1)
# ============================================================================

print("\n" + "="*80)
print("🔬 STEP 9: SHAP VALUES - MODEL EXPLAINABILITY (Phase 1 CRITICAL)")
print("="*80)

print("Computing SHAP values (this may take a few minutes for large datasets)...")

# Use a sample for SHAP if dataset is very large (>10k samples)
X_test_shap = X_test
if len(X_test) > 10000:
    print(f"⚠️  Sampling {10000} test samples for SHAP computation...")
    X_test_shap = X_test.sample(n=10000, random_state=RANDOM_STATE)

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test_shap)

print("✅ SHAP values computed successfully!")

# 9.1: SHAP Summary Plot (Bar)
print("\n📊 Generating SHAP summary plot (bar)...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_shap, plot_type="bar", show=False)
plt.tight_layout()
shap_bar_path = os.path.join(PLOTS_DIR, 'xgboost_shap_summary_bar.png')
plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
print(f"✅ SHAP summary bar plot saved")
tracker.log_figure(plt.gcf(), 'shap_summary_bar.png')
plt.close()

# 9.2: SHAP Summary Plot (Beeswarm)
print("\n📊 Generating SHAP beeswarm plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_shap, show=False)
plt.tight_layout()
shap_beeswarm_path = os.path.join(PLOTS_DIR, 'xgboost_shap_beeswarm.png')
plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches='tight')
print(f"✅ SHAP beeswarm plot saved")
tracker.log_figure(plt.gcf(), 'shap_beeswarm.png')
plt.close()

# 9.3: Export SHAP values (optional - for dashboard)
shap_df = pd.DataFrame(shap_values.values, columns=X_test_shap.columns)
shap_df['actual_popularity'] = y_test.loc[X_test_shap.index].values
shap_df['predicted_popularity'] = model.predict(X_test_shap)
shap_export_path = os.path.join(OUTPUTS_DIR, 'shap_values_per_track.csv')
shap_df.to_csv(shap_export_path, index=False)
print(f"✅ SHAP values exported to: {shap_export_path}")


# ============================================================================
# STEP 10: SAVE MODEL & METADATA (PHASE 1)
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 10: SAVE MODEL & METADATA (Phase 1 Improvement)")
print("="*80)

# Create comprehensive metadata
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metadata = create_model_metadata(
    model_params=params,
    metrics=metrics,
    feature_names=audio_features,
    train_size=X_train.shape,
    test_size=X_test.shape
)

# Save model and metadata
model_path = os.path.join(MODELS_DIR, f'xgb_model_{timestamp}.joblib')
metadata_path = os.path.join(METADATA_DIR, f'xgb_metadata_{timestamp}.json')

save_model_with_metadata(model, metadata, model_path, metadata_path)

# Log model to MLflow
print("\n📊 Logging model to MLflow...")
tracker.log_model(
    model,
    artifact_path="model",
    registered_model_name="spotify_xgboost_model"
)

# Log additional artifacts
tracker.log_artifact(metadata_path, "metadata")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✨ PIPELINE COMPLETE!")
print("="*80)

print("\n📊 Final Test Set Performance:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  R²:   {test_r2:.4f}")
print(f"  Adjusted R²: {test_adj_r2:.4f}")

print("\n📁 Outputs saved to:")
print(f"  Plots:    {PLOTS_DIR}/")
print(f"  Model:    {model_path}")
print(f"  Metadata: {metadata_path}")

print("\n🎯 Key Improvements Implemented:")
print("  ✅ Phase 1: Data validation & separate validation set")
print("  ✅ Phase 1: Git commit hash & environment metadata")
print("  ✅ Phase 1 CRITICAL: SHAP values for model interpretability")
print("  ✅ Phase 2: JSON config file for reproducibility")
print("  ✅ Phase 2: Learning curves for overfitting detection")
print("  ✅ Phase 2: Adjusted R² metric")
print("  ✅ Phase 2: Correlation heatmap & QQ plots")
print("  ✅ Phase 3: Full MLflow experiment tracking")

print(f"\n🔍 View results in MLflow UI:")
print(f"   Run: make mlflow-ui")
print(f"   Or:  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db")

print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# End MLflow run
tracker.end_run(status="FINISHED")
