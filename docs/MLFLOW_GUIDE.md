# MLflow Integration Guide

This guide explains how to use MLflow for experiment tracking in the Spotify Track Analytics project.

## Overview

MLflow integration provides:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage trained models
- **UI Dashboard**: Visual comparison of experiment runs
- **SQLite Backend**: Local storage (no server required)

## Quick Start

### 1. Start MLflow UI

```bash
make mlflow-ui
```

Then open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Run Experiment with MLflow Tracking

```python
from src.mlflow_tracker import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(
    experiment_name="spotify_popularity_prediction"
)

# Start run
run = tracker.start_run(run_name="my_experiment")

# Log parameters
tracker.log_params({
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200
})

# Train model
model = XGBRegressor(**params)
model.fit(X_train, y_train)

# Log metrics
tracker.log_metrics({
    'test_rmse': test_rmse,
    'test_r2': test_r2
})

# Log model
tracker.log_model(model, registered_model_name="spotify_xgboost_model")

# End run
tracker.end_run()
```

### 3. View Results

```bash
# View all experiments
make mlflow-info

# Export to CSV
make mlflow-export

# Open UI
make mlflow-ui
```

## MLflow Directory Structure

```
├── mlruns/                  # Experiment metadata
│   └── mlflow.db           # SQLite database
├── mlartifacts/            # Artifacts (models, plots, etc.)
│   └── 1/                  # Experiment ID
│       └── <run_id>/       # Run artifacts
└── exports/                # Exported CSV files
```

## Configuration

### SQLite Backend (Default)

The project uses SQLite for local experiment tracking:

```python
tracking_uri = "sqlite:///mlruns/mlflow.db"
```

**Benefits**:
- No server setup required
- All data stored locally
- Fast and lightweight
- Easy to backup (just copy `mlruns/` and `mlartifacts/`)

### Changing Backend (Optional)

To use a remote tracking server:

```python
tracker = MLflowTracker(
    experiment_name="spotify_popularity_prediction",
    tracking_uri="http://your-mlflow-server:5000"
)
```

## Using the MLflow Tracker

### Basic Usage

```python
from src.mlflow_tracker import MLflowTracker

# Initialize
tracker = MLflowTracker()

# Start run
run = tracker.start_run(
    run_name="experiment_1",
    tags={'model_type': 'xgboost', 'version': 'v1'}
)

# Log everything
tracker.log_params({'max_depth': 6})
tracker.log_metrics({'rmse': 5.2, 'r2': 0.85})
tracker.log_model(model)
tracker.log_artifact('plots/learning_curve.png')

# End run
tracker.end_run()
```

### Convenience Function

For quick experiments:

```python
from src.mlflow_tracker import track_experiment

run_id = track_experiment(
    experiment_name="spotify_popularity_prediction",
    model=trained_model,
    params={'max_depth': 6, 'learning_rate': 0.1},
    metrics={'test_rmse': 5.2, 'test_r2': 0.85},
    artifacts={'plots': 'outputs/plots/'},
    run_name="quick_experiment",
    registered_model_name="spotify_xgboost_model"
)
```

## Logging Different Types of Data

### 1. Parameters (Hyperparameters)

```python
tracker.log_params({
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8
})
```

### 2. Metrics (Evaluation Results)

```python
tracker.log_metrics({
    'train_rmse': 4.5,
    'val_rmse': 5.1,
    'test_rmse': 5.2,
    'test_r2': 0.85,
    'test_adjusted_r2': 0.84
})
```

### 3. Models

```python
# Log XGBoost model
tracker.log_model(
    model,
    registered_model_name="spotify_xgboost_model"
)

# Model is automatically versioned in registry
```

### 4. Artifacts (Files)

```python
# Single file
tracker.log_artifact('outputs/plots/learning_curve.png')

# Entire directory
tracker.log_artifacts('outputs/plots/', artifact_path='plots')

# Matplotlib figure
tracker.log_figure(fig, 'confusion_matrix.png')

# Dictionary as JSON
tracker.log_dict(metadata, 'metadata.json')

# Text content
tracker.log_text(report, 'report.txt')
```

### 5. Tags (Metadata)

```python
tracker.set_tags({
    'model_type': 'xgboost',
    'dataset': 'spotify_114k',
    'pipeline_version': 'v1.0',
    'git_commit': 'abc123'
})
```

## Comparing Experiments

### In MLflow UI

1. Start UI: `make mlflow-ui`
2. Select multiple runs (checkbox)
3. Click "Compare"
4. View side-by-side comparison of:
   - Parameters
   - Metrics
   - Artifacts (plots, models)

### Programmatically

```python
from src.mlflow_tracker import MLflowTracker

# Search runs
runs = MLflowTracker.search_runs(
    experiment_names=["spotify_popularity_prediction"],
    filter_string="metrics.test_r2 > 0.80",
    order_by=["metrics.test_rmse ASC"]
)

# Display results
print(runs[['run_id', 'metrics.test_r2', 'metrics.test_rmse', 'params.max_depth']])
```

## Model Registry

### Registering Models

Models are automatically registered when you use `registered_model_name`:

```python
tracker.log_model(
    model,
    registered_model_name="spotify_xgboost_model"
)
```

### Model Versions

Each registration creates a new version:
- Version 1, 2, 3, etc.
- Each version linked to a specific run
- Can promote versions to stages: "Staging", "Production"

### Loading Registered Models

```python
import mlflow.xgboost

# Load latest version
model = mlflow.xgboost.load_model("models:/spotify_xgboost_model/latest")

# Load specific version
model = mlflow.xgboost.load_model("models:/spotify_xgboost_model/2")

# Load from production stage
model = mlflow.xgboost.load_model("models:/spotify_xgboost_model/Production")
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make mlflow-ui` | Start MLflow UI at http://127.0.0.1:5000 |
| `make mlflow-info` | Show experiment statistics |
| `make mlflow-export` | Export runs to `exports/mlflow_runs.csv` |
| `make mlflow-clean` | Delete all experiments and artifacts |
| `make train-mlflow` | Run pipeline with MLflow tracking |

## Best Practices

### 1. Consistent Naming

Use clear, descriptive names:
```python
run_name = f"xgboost_{max_depth}d_{learning_rate}lr"
```

### 2. Tag Everything

```python
tags = {
    'model_type': 'xgboost',
    'feature_set': 'audio_features_v2',
    'data_version': '2025-01-13',
    'git_commit': get_git_commit(),
    'purpose': 'hyperparameter_tuning'
}
```

### 3. Log Comprehensive Metrics

```python
metrics = {
    # Standard metrics
    'train_rmse': train_rmse,
    'val_rmse': val_rmse,
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'test_adjusted_r2': test_adj_r2,
    
    # Additional metrics
    'train_time_seconds': training_time,
    'n_estimators_used': model.best_iteration,
    'overfitting_score': train_r2 - test_r2
}
```

### 4. Log Important Artifacts

```python
# Learning curves
tracker.log_artifact('outputs/plots/learning_curve.png')

# SHAP plots
tracker.log_artifact('outputs/plots/shap_summary.png')

# Feature importance
tracker.log_artifact('outputs/models/feature_importance.csv')

# Model metadata
tracker.log_dict(metadata, 'model_metadata.json')
```

### 5. Use Run Context Manager (Advanced)

```python
with mlflow.start_run(run_name="my_experiment") as run:
    mlflow.log_params(params)
    # Train model
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "model")
    # Automatically ends run
```

## Troubleshooting

### Issue: "SQLAlchemy engine could not be created"

**Solution**: Ensure `mlruns/` directory exists:
```bash
mkdir -p mlruns mlartifacts
```

### Issue: UI shows no experiments

**Solution**: Check tracking URI matches:
```bash
# In code
tracking_uri = "sqlite:///mlruns/mlflow.db"

# When starting UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Issue: Runs not appearing in UI

**Solution**: Refresh the UI or restart:
```bash
# Kill existing MLflow UI process
pkill -f "mlflow ui"

# Restart
make mlflow-ui
```

### Issue: Large artifacts causing slow tracking

**Solution**: Only log essential artifacts:
```python
# Instead of logging all plots
tracker.log_artifacts('outputs/plots/')  # Logs everything

# Log selectively
tracker.log_artifact('outputs/plots/shap_summary.png')  # Only important ones
```

## Advanced Features

### Nested Runs

For hyperparameter tuning:

```python
with mlflow.start_run(run_name="hyperparameter_search"):
    for max_depth in [4, 6, 8]:
        with mlflow.start_run(run_name=f"depth_{max_depth}", nested=True):
            # Train and log
            pass
```

### Autologging (Automatic Logging)

```python
import mlflow.xgboost

# Enable autologging
mlflow.xgboost.autolog()

# Now MLflow automatically logs:
# - Parameters
# - Metrics
# - Model
# - Feature importance

model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### Custom Metrics Over Time

```python
for iteration in range(n_iterations):
    # Train one epoch
    current_loss = train_epoch()
    
    # Log metric with step
    tracker.log_metrics({'loss': current_loss}, step=iteration)
```

## Integration with Existing Pipeline

The improved ML pipeline can be enhanced with MLflow:

```python
# At the start of your pipeline
tracker = MLflowTracker()
run = tracker.start_run(run_name="improved_pipeline_run")

# Log configuration
tracker.log_params(load_config('config/xgboost_params.json'))

# After training
tracker.log_metrics(metrics)
tracker.log_model(model, registered_model_name="spotify_xgboost_model")

# Log all plots
tracker.log_artifacts('outputs/plots/', artifact_path='visualizations')

# Log SHAP values
tracker.log_artifact('outputs/shap_values_per_track.csv', artifact_path='interpretability')

# End run
tracker.end_run()
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)

---

**Status**: ✅ MLflow Integration Complete

**Questions?** See `src/mlflow_tracker.py` for implementation details.
