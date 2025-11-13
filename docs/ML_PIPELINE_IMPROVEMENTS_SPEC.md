# ML Pipeline Improvements - Technical Specification

**Project**: Spotify Track Analytics Popularity Prediction
**Document Version**: 1.0
**Date**: 2025-11-13
**Status**: Draft - Pending Implementation

---

## Executive Summary

This specification document outlines improvements to the XGBoost-based machine learning pipeline for predicting Spotify track popularity. The recommendations are organized into **Core Improvements** (essential for production readiness) and **Advanced Enhancements** (optional optimizations for scaling).

These improvements address:
- Data validation and reproducibility
- Model interpretability (SHAP analysis)
- Training diagnostics and overfitting detection
- Experiment tracking and versioning
- Advanced hyperparameter optimization

---

## Table of Contents

1. [Data Loading & Validation](#1-data-loading--validation)
2. [Model Initialization](#2-model-initialization)
3. [Model Training](#3-model-training)
4. [Predictions & Evaluation](#4-predictions--evaluation)
5. [Visualizations](#5-visualizations)
6. [Feature Importance & Explainability](#6-feature-importance--explainability)
7. [Model Saving & Metadata](#7-model-saving--metadata)
8. [Future Enhancements (Optional)](#8-future-enhancements-optional)

---

## 1. Data Loading & Validation

### Current State
✅ **Already Good**: Loads data from Parquet format (efficient storage).

### Improvements Required

#### 1.1 Data Sanity Checks
**Priority**: HIGH
**Rationale**: Prevent silent failures from data quality issues.

**Implementation**:
```python
# Feature consistency validation
assert list(X_train.columns) == list(X_test.columns), "❌ Feature mismatch between train/test sets"
print(f"✅ Features consistent across datasets.")

# Optional: Missing values check
print(f"Missing values in training data: {X_train.isnull().sum().sum()}")
print(f"Missing values in test data: {X_test.isnull().sum().sum()}")
```

**Acceptance Criteria**:
- Assert statement confirms column alignment between train/test splits
- Optional logging of missing value counts for debugging
- Fails fast if feature mismatch detected

**Dependencies**: None
**Estimated Effort**: 0.5 hours

---

## 2. Model Initialization

### Current State
✅ **Great**: Uses sensible default XGBoost parameters for baseline training.

### Improvements Required

#### 2.1 JSON Configuration File
**Priority**: MEDIUM
**Rationale**: Improve reproducibility and parameter management across experiments.

**Implementation**:
```python
import json

# config/xgboost_params.json
{
  "max_depth": 6,
  "learning_rate": 0.1,
  "n_estimators": 100,
  "objective": "reg:squarederror",
  "random_state": 42
}

# In training script
with open("../config/xgboost_params.json") as f:
    params = json.load(f)
model = XGBRegressor(**params)
```

**Acceptance Criteria**:
- Config file stores all hyperparameters in JSON format
- Model loads parameters from config file
- Config file is version controlled for experiment tracking

**Dependencies**: Create `config/` directory
**Estimated Effort**: 1 hour

#### 2.2 Learning Rate Scheduler
**Priority**: MEDIUM
**Rationale**: Improve convergence on large datasets (114K tracks).

**Implementation**:
```python
# Add to XGBoost params for adaptive learning rate
callbacks = [
    xgb.callback.EarlyStopping(rounds=10, metric_name='rmse',
                                data_name='validation_0', save_best=True)
]

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=callbacks,
          verbose=True)
```

**Acceptance Criteria**:
- Early stopping prevents overfitting on validation set
- Learning rate adapts based on validation performance
- Best model is automatically saved

**Dependencies**: Separate validation set (see 3.1)
**Estimated Effort**: 1.5 hours

#### 2.3 Parameter Logging
**Priority**: LOW
**Rationale**: Enables experiment comparison and reproducibility.

**Implementation**:
```python
# Log to MLflow (if integrated) or simple logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Training XGBoost with params: {params}")
logger.info(f"Training set size: {X_train.shape}")
```

**Acceptance Criteria**:
- All hyperparameters logged before training
- Training set dimensions logged
- Logs stored in `logs/` directory with timestamps

**Dependencies**: None
**Estimated Effort**: 1 hour

---

## 3. Model Training

### Current State
✅ **Good**: Includes early stopping and basic evaluation metrics.

### Improvements Required

#### 3.1 Separate Validation Set
**Priority**: HIGH
**Rationale**: Current implementation may reuse test data for validation, causing data leakage.

**Implementation**:
```python
from sklearn.model_selection import train_test_split

# Split training data into train + validation
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Now we have: 64% train, 16% validation, 20% test
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

**Acceptance Criteria**:
- Three distinct datasets: train (64%), validation (16%), test (20%)
- No data leakage between sets
- Test set remains untouched until final evaluation

**Dependencies**: None
**Estimated Effort**: 1 hour

#### 3.2 Learning Curves Visualization
**Priority**: MEDIUM
**Rationale**: Diagnose overfitting and validate model convergence.

**Implementation**:
```python
results = model.evals_result()

plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['rmse'], label='Train RMSE')
plt.plot(results['validation_1']['rmse'], label='Validation RMSE')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/plots/xgboost_learning_curve.png', dpi=300)
plt.show()
```

**Acceptance Criteria**:
- Plot shows both training and validation RMSE over iterations
- Visual confirmation of convergence (curves plateau)
- Saved to `outputs/plots/` directory with timestamp

**Dependencies**: Section 3.1 (validation set)
**Estimated Effort**: 1.5 hours

---

## 4. Predictions & Evaluation

### Current State
✅ **Good**: Clean evaluation logic with standard regression metrics.

### Improvements Required

#### 4.1 Adjusted R² Metric
**Priority**: MEDIUM
**Rationale**: Better metric for multivariate model comparison (accounts for number of features).

**Implementation**:
```python
def adjusted_r2(r2, n, k):
    """
    Calculate adjusted R² score.

    Args:
        r2: R² score from model
        n: Number of samples
        k: Number of features

    Returns:
        Adjusted R² score
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# In evaluation section
r2_score_val = r2_score(y_test, y_test_pred)
adj_r2 = adjusted_r2(r2_score_val, len(y_test), X_test.shape[1])
print(f"Adjusted R²: {adj_r2:.4f}")
```

**Acceptance Criteria**:
- Function computes adjusted R² correctly
- Metric logged alongside standard R²
- Difference between R² and adjusted R² documented

**Dependencies**: None
**Estimated Effort**: 0.5 hours

#### 4.2 Scatter Density Plots
**Priority**: LOW
**Rationale**: Better visualization for dense prediction regions (114K samples).

**Implementation**:
```python
import seaborn as sns

# Use seaborn's kdeplot for density visualization
plt.figure(figsize=(8, 6))
sns.kdeplot(x=y_test, y=y_test_pred, cmap="Blues", fill=True, thresh=0.05)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Prediction Density Plot')
plt.legend()
plt.savefig('../outputs/plots/prediction_density.png', dpi=300)
plt.show()
```

**Acceptance Criteria**:
- Density plot shows concentration of predictions
- Easier to interpret than standard scatter for large datasets
- Saved to `outputs/plots/` directory

**Dependencies**: Install `seaborn` (already in requirements.txt)
**Estimated Effort**: 1 hour

---

## 5. Visualizations

### Current State
✅ **Comprehensive**: Includes actual vs predicted plots and residual analysis.

### Improvements Required

#### 5.1 Correlation Heatmap
**Priority**: MEDIUM
**Rationale**: Extend EDA to understand feature-target relationships.

**Implementation**:
```python
# Combine features + target for correlation analysis
analysis_df = X_train.copy()
analysis_df['popularity'] = y_train

plt.figure(figsize=(12, 10))
correlation_matrix = analysis_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap (with Target)')
plt.tight_layout()
plt.savefig('../outputs/plots/correlation_heatmap.png', dpi=300)
plt.show()
```

**Acceptance Criteria**:
- Heatmap shows correlations between all features and target
- Annotations display correlation coefficients
- Saved to `outputs/plots/` for EDA documentation

**Dependencies**: None
**Estimated Effort**: 1 hour

#### 5.2 SHAP Summary Plot
**Priority**: HIGH (See Section 6)
**Rationale**: Critical for model interpretability.

**Details**: See Section 6.1 (Feature Importance & Explainability)

#### 5.3 QQ Plot for Residuals
**Priority**: LOW
**Rationale**: Validate normality assumption for residuals.

**Implementation**:
```python
import scipy.stats as stats

residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.tight_layout()
plt.savefig('../outputs/plots/qq_plot_residuals.png', dpi=300)
plt.show()
```

**Acceptance Criteria**:
- QQ plot confirms approximate normality of residuals
- Deviations from normality documented
- Saved to `outputs/plots/` directory

**Dependencies**: None
**Estimated Effort**: 0.5 hours

---

## 6. Feature Importance & Explainability

### Current State
✅ **Fine**: Uses built-in `model.feature_importances_` for basic importance ranking.

### Major Improvement Required

#### 6.1 SHAP Values (Model Explainability)
**Priority**: CRITICAL
**Rationale**: Essential for understanding model decisions, identifying feature interactions, and building trust with stakeholders.

**Implementation**:
```python
import shap

# Create SHAP explainer for XGBoost
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot (most important)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('../outputs/plots/xgboost_shap_summary.png', dpi=300)
plt.show()

# Detailed beeswarm plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('../outputs/plots/xgboost_shap_beeswarm.png', dpi=300)
plt.show()
```

**Acceptance Criteria**:
- SHAP summary bar plot shows global feature importance
- SHAP beeswarm plot shows feature impact distributions
- Both plots saved to `outputs/plots/` directory
- Optional: Export SHAP values per track for dashboard integration

**Dependencies**:
- Install `shap` library: `pip install shap`
- Requires trained model

**Estimated Effort**: 2-3 hours

**Advanced Option**:
```python
# Export SHAP values per track for Streamlit dashboard
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
shap_df['track_name'] = test_data['track_name'].values  # If available
shap_df.to_csv('../outputs/shap_values_per_track.csv', index=False)
```

---

## 7. Model Saving & Metadata

### Current State
✅ **Excellent**: Saves model, parameters, and performance metadata using `joblib`.

### Improvements Required

#### 7.1 Git Commit Hash
**Priority**: HIGH
**Rationale**: Critical for reproducibility - links model to exact codebase version.

**Implementation**:
```python
import subprocess

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    metadata['git_commit'] = git_commit
except Exception:
    metadata['git_commit'] = None

print(f"Model trained on git commit: {metadata['git_commit']}")
```

**Acceptance Criteria**:
- Git commit hash stored in model metadata
- Graceful handling if not in git repository
- Metadata saved alongside model file

**Dependencies**: Git repository initialized
**Estimated Effort**: 0.5 hours

#### 7.2 Python & Library Versions
**Priority**: HIGH
**Rationale**: Ensure reproducibility across different environments.

**Implementation**:
```python
import platform
import sys
import xgboost

metadata['environment'] = {
    'python_version': sys.version,
    'platform': platform.platform(),
    'xgboost_version': xgboost.__version__,
    'scikit_learn_version': sklearn.__version__,
    'pandas_version': pd.__version__
}

print(f"Environment: {metadata['environment']}")
```

**Acceptance Criteria**:
- Python version, OS, and key library versions stored
- Metadata includes all critical dependencies
- Version info accessible for debugging

**Dependencies**: None
**Estimated Effort**: 1 hour

**File Structure**:
```
outputs/
├── models/
│   └── xgb_model_{timestamp}.joblib
├── metadata/
│   └── xgb_metadata_{timestamp}.json
└── plots/
    └── (all visualization outputs)
```

---

## 8. Future Enhancements (Optional)

### 8.1 Automated Hyperparameter Tuning
**Priority**: LOW
**Rationale**: Optimize performance beyond manual tuning. High computational cost.

**Recommended Tools**:
- **Optuna** (preferred): Bayesian optimization, visualization support
- **GridSearchCV**: Exhaustive search for smaller parameter spaces

**Implementation Example (Optuna)**:
```python
from sklearn.model_selection import RandomizedSearchCV

params = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 400, 600],
}

search = RandomizedSearchCV(
    model, params, n_iter=10, cv=3,
    scoring='r2', verbose=2, n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

**Acceptance Criteria**:
- Automated search runs with specified parameter grid
- Best parameters logged and saved to config
- Cross-validation prevents overfitting

**Dependencies**: Large compute resources (recommend cloud instance)
**Estimated Effort**: 4-6 hours

---

### 8.2 Model Versioning
**Priority**: MEDIUM
**Rationale**: Enable A/B testing and rollback capability.

**Implementation**:
```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"../outputs/models/xgb_model_{timestamp}.joblib"
joblib.dump(model, model_path)
print(f"Model saved: {model_path}")
```

**Acceptance Criteria**:
- Models named with timestamps or performance metrics
- Metadata links model file to experiment parameters
- Directory structure supports multiple model versions

**Dependencies**: None
**Estimated Effort**: 1 hour

---

### 8.3 MLflow / Weights & Biases Integration
**Priority**: LOW
**Rationale**: Production-grade experiment tracking. Requires infrastructure setup.

**Recommended Tool**: **MLflow** (open-source, self-hosted)

**Implementation Overview**:
```python
import mlflow
import mlflow.xgboost

mlflow.set_experiment("spotify_popularity_prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("adjusted_r2", adj_r2)

    # Log model
    mlflow.xgboost.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("../outputs/plots/learning_curve.png")
```

**Acceptance Criteria**:
- All experiments tracked in MLflow UI
- Models, metrics, and artifacts logged automatically
- Comparison dashboard available for stakeholders

**Dependencies**:
- MLflow installation: `pip install mlflow`
- Server setup (local or cloud)

**Estimated Effort**: 6-8 hours (including setup)

---

## Implementation Roadmap

### Phase 1: Critical Improvements (Week 1)
**Goal**: Improve data integrity, model interpretability, and reproducibility.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| 1.1 Data Sanity Checks | HIGH | 0.5h | None |
| 3.1 Separate Validation Set | HIGH | 1h | None |
| 6.1 SHAP Values | CRITICAL | 2-3h | Trained model |
| 7.1 Git Commit Hash | HIGH | 0.5h | Git repo |
| 7.2 Python/Library Versions | HIGH | 1h | None |

**Total Effort**: 5-6 hours

---

### Phase 2: Enhanced Diagnostics (Week 2)
**Goal**: Improve training visibility and evaluation metrics.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| 2.1 JSON Config File | MEDIUM | 1h | None |
| 3.2 Learning Curves | MEDIUM | 1.5h | Section 3.1 |
| 4.1 Adjusted R² | MEDIUM | 0.5h | None |
| 5.1 Correlation Heatmap | MEDIUM | 1h | None |

**Total Effort**: 4 hours

---

### Phase 3: Advanced Features (Optional)
**Goal**: Production-grade experiment tracking and optimization.

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| 2.2 Learning Rate Scheduler | MEDIUM | 1.5h | Section 3.1 |
| 8.1 Hyperparameter Tuning | LOW | 4-6h | Compute resources |
| 8.3 MLflow Integration | LOW | 6-8h | MLflow setup |

**Total Effort**: 12-16 hours

---

## Dependencies & Prerequisites

### Python Libraries (Add to requirements.txt)
```txt
# Core ML (already present)
xgboost>=1.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# New additions for improvements
shap>=0.42.0          # For Section 6.1 (SHAP values)
seaborn>=0.12.0       # For Section 4.2 (density plots)
mlflow>=2.7.0         # Optional: Section 8.3 (experiment tracking)
optuna>=3.3.0         # Optional: Section 8.1 (hyperparameter tuning)
```

### Directory Structure
```bash
mkdir -p config                    # For Section 2.1
mkdir -p outputs/models            # Already exists
mkdir -p outputs/metadata          # For Section 7
mkdir -p outputs/plots             # Already exists
mkdir -p logs                      # For Section 2.3
```

---

## Testing & Validation

### Unit Tests
Create `tests/test_improvements.py`:

```python
import pytest
from src.utils import adjusted_r2

def test_adjusted_r2():
    """Test adjusted R² calculation."""
    # Perfect model: R² = 1
    assert adjusted_r2(r2=1.0, n=100, k=10) == 1.0

    # Adjusted R² should be lower than R² for same model
    r2_val = 0.85
    adj_r2 = adjusted_r2(r2=r2_val, n=100, k=10)
    assert adj_r2 < r2_val

def test_data_validation():
    """Test feature consistency check."""
    import pandas as pd

    train_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    test_df = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

    assert list(train_df.columns) == list(test_df.columns)
```

### Integration Tests
- Run full pipeline with small dataset (1000 samples)
- Verify all outputs (plots, metadata, models) are generated
- Confirm SHAP values run without errors

---

## Success Metrics

### Quantitative Goals
- **Reproducibility**: 100% of experiments reproducible via git commit hash + environment metadata
- **Interpretability**: SHAP plots generated for all models (100% coverage)
- **Model Performance**: Adjusted R² within 5% of standard R²
- **Training Diagnostics**: Learning curves show convergence within 100 iterations

### Qualitative Goals
- Stakeholders can interpret model decisions via SHAP plots
- Data scientists can reproduce any past experiment
- Codebase follows best practices for ML pipelines

---

## Open Questions & Risks

### Questions for Stakeholders
1. **MLflow Setup**: Do we have infrastructure for self-hosted MLflow server?
2. **Compute Resources**: Is hyperparameter tuning budget approved? (Optuna can be compute-intensive)
3. **Dashboard Integration**: Should SHAP values be exposed in Streamlit app?

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| SHAP computation time on 114K samples | HIGH | Use TreeExplainer (optimized for XGBoost), or sample 10K for visualization |
| Breaking existing notebooks | MEDIUM | Create new `Hackathon2Music_v2.ipynb` for improvements |
| Dependency conflicts | LOW | Use virtual environment, pin versions in requirements.txt |

---

## References

### Documentation
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Related Files
- Notebook: `notebooks/Hackathon2Music.ipynb`
- Requirements: `requirements.txt`
- Project README: `README.md`
- Project guidance: `.claude/CLAUDE.md`

---

## Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-13 | Collaborator + LLM | Initial specification based on 8 improvement suggestions |

---

## Appendix: Code Snippets by Priority

### HIGH Priority (Must Implement)
1. Data validation (Section 1.1)
2. Separate validation set (Section 3.1)
3. SHAP values (Section 6.1)
4. Git commit hash (Section 7.1)
5. Environment versioning (Section 7.2)

### MEDIUM Priority (Recommended)
6. JSON config (Section 2.1)
7. Learning curves (Section 3.2)
8. Adjusted R² (Section 4.1)
9. Correlation heatmap (Section 5.1)
10. Model versioning (Section 8.2)

### LOW Priority (Nice to Have)
11. Parameter logging (Section 2.3)
12. Scatter density plots (Section 4.2)
13. QQ plots (Section 5.3)
14. Hyperparameter tuning (Section 8.1)
15. MLflow integration (Section 8.3)

---

**End of Specification**
