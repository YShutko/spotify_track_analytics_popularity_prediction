# Makefile Commands Reference

Complete reference for all available `make` commands in the Spotify Track Analytics project.

## Quick Start

```bash
# Show all available commands
make help

# Run full pipeline with synthetic data (testing)
make full-pipeline

# Run with real data (requires ETL)
make full-pipeline-real
```

---

## General Commands

### `make help`
Display help message with all available commands.

### `make setup`
Install all dependencies from requirements.txt.

```bash
make setup
```

### `make clean`
Clean generated files and caches:
- `__pycache__` directories
- `.ipynb_checkpoints`
- `*.pyc`, `*.pyo` files
- `.DS_Store` files

```bash
make clean
```

---

## MLflow Commands

### `make mlflow-ui`
Start MLflow UI with SQLite backend at http://127.0.0.1:5000

```bash
make mlflow-ui
```

### `make mlflow-info`
Show MLflow experiment statistics (experiments, runs count).

```bash
make mlflow-info
```

**Example Output:**
```
📊 MLflow Information:
   Tracking URI: sqlite:///mlruns/mlflow.db
   Artifacts: ./mlartifacts
   Total Experiments: 1
   - spotify_popularity_prediction: 3 runs
```

### `make mlflow-export`
Export all MLflow runs to `exports/mlflow_runs.csv`.

```bash
make mlflow-export
```

### `make mlflow-clean`
Delete all MLflow experiments, runs, and artifacts (prompts for confirmation).

```bash
make mlflow-clean
```

### `make mlflow-reset`
Reset corrupted MLflow database by deleting and recreating it.

```bash
make mlflow-reset
```

**What it does:**
- Stops any running MLflow UI processes
- Deletes corrupted database files (mlflow.db, mlflow.db-shm, mlflow.db-wal)
- Recreates directory structure
- Database will be created fresh on next start

**When to use:**
- MLflow UI fails to start with Alembic migration errors
- Database reports "inconsistent state"
- After upgrading MLflow version

### `make mlflow-stop`
Cleanly stop MLflow UI server and all worker processes.

```bash
make mlflow-stop
```

---

## Training Commands

### `make train`
Run improved ML pipeline (requires `cleaned_music_data.csv`).

```bash
make train
```

### `make train-mlflow`
Run improved ML pipeline with MLflow tracking enabled.

```bash
make train-mlflow
```

### `make test-pipeline`
Test pipeline with synthetic data (generates 1,000 samples).

```bash
make test-pipeline
```

### `make prepare-data`
Convert processed parquet files to `cleaned_music_data.csv`.

```bash
make prepare-data
```

**Requirements:**
- `data/processed/ml_ready_data.parquet` must exist

---

## Notebook Commands

### `make notebook`
Start Jupyter notebook server.

```bash
make notebook
```

### `make notebook-improved`
Open improved ML pipeline notebook directly.

```bash
make notebook-improved
```

---

## Dashboard Commands

### `make streamlit`
Run Streamlit dashboard at http://localhost:8501

```bash
make streamlit
```

**Features:**
- Interactive model predictions
- Feature importance visualization
- SHAP explanations
- Track search and filtering

### `make gradio`
Run Gradio dashboard at http://localhost:7860

```bash
make gradio
```

**Features:**
- Simple web interface
- Real-time predictions
- Easy to share

### `make hf-space`
Run Hugging Face Space app locally.

```bash
make hf-space
```

---

## Development Commands

### `make lint`
Run quick code quality checks with flake8.

```bash
make lint
```

**Settings:**
- Max line length: 100
- Ignores: E203, W503

### `make lint-all`
Run comprehensive linting (flake8, pylint, mypy).

```bash
make lint-all
```

**Checks:**
1. **flake8**: Style and syntax
2. **pylint**: Code quality and bugs
3. **mypy**: Type checking

### `make format`
Format source code with black.

```bash
make format
```

**Applies to:** `src/` directory only

### `make format-all`
Format all Python files (src + apps).

```bash
make format-all
```

**Applies to:**
- `src/`
- `app.py` (Streamlit)
- `app_gradio.py` (Gradio)

---

## Git Commands

### `make git-status`
Show enhanced git status with recent commits.

```bash
make git-status
```

**Output includes:**
- Current branch and tracking info
- Files changed
- Last 5 commits

### `make git-push`
Push commits to origin/main.

```bash
make git-push
```

---

## Output Commands

### `make view-plots`
Open outputs/plots directory in file browser.

```bash
make view-plots
```

### `make view-models`
List saved models with file sizes.

```bash
make view-models
```

**Example Output:**
```
💾 Saved Models:
-rw-r--r--  1 user  staff   682K xgb_model_20251113_092654.joblib
-rw-r--r--  1 user  staff   934K xgboost_popularity_model.joblib
```

### `make view-metadata`
Show model metadata files and display latest.

```bash
make view-metadata
```

---

## Documentation Commands

### `make docs`
List all project documentation files.

```bash
make docs
```

**Output:**
```
📚 Project Documentation:
   - Full Spec: docs/ML_PIPELINE_IMPROVEMENTS_SPEC.md
   - Implementation Summary: docs/IMPLEMENTATION_SUMMARY.md
   - Quick Start: docs/README_IMPROVEMENTS.md
   - MLflow Guide: docs/MLFLOW_GUIDE.md
```

### `make view-spec`
View ML pipeline improvements specification in less.

```bash
make view-spec
```

---

## Docker Commands (Optional)

### `make docker-build`
Build Docker image for the project.

```bash
make docker-build
```

**Image name:** `spotify-ml-pipeline`

### `make docker-run`
Run Docker container with port mappings.

```bash
make docker-run
```

**Ports:**
- 5000: MLflow UI
- 8888: Jupyter notebook

---

## All-in-One Commands

### `make full-pipeline`
Run complete pipeline with synthetic data.

```bash
make full-pipeline
```

**Steps:**
1. Clean caches
2. Install dependencies
3. Generate synthetic data
4. Train model
5. Generate visualizations

**Use for:** Testing, development

### `make full-pipeline-real`
Run complete pipeline with real data.

```bash
make full-pipeline-real
```

**Steps:**
1. Clean caches
2. Install dependencies
3. Prepare data from parquet
4. Train model
5. Generate visualizations

**Use for:** Production training

**Requirements:**
- `data/processed/ml_ready_data.parquet` must exist
- Run ETL pipeline first if needed

---

## Command Chaining

You can chain multiple commands:

```bash
# Clean, setup, and train
make clean setup train

# Format, lint, and push
make format lint git-push

# Train and view results
make train && make view-plots && make mlflow-ui
```

---

## Common Workflows

### Development Workflow

```bash
# 1. Setup environment
make setup

# 2. Test with synthetic data
make test-pipeline

# 3. View results
make view-plots
make mlflow-ui

# 4. Format and lint code
make format lint
```

### Production Training Workflow

```bash
# 1. Prepare data
make prepare-data

# 2. Train model
make train-mlflow

# 3. View results in MLflow
make mlflow-ui

# 4. Export experiment data
make mlflow-export
```

### Dashboard Deployment Workflow

```bash
# 1. Ensure model is trained
make train

# 2. Run Streamlit dashboard
make streamlit

# Or run Gradio
make gradio
```

### Code Quality Workflow

```bash
# 1. Format code
make format-all

# 2. Run linting
make lint-all

# 3. Commit and push
make git-push
```

---

## Troubleshooting

### Issue: `make train` fails with "No such file"

**Solution:** Run `make test-pipeline` or `make prepare-data` first:

```bash
# Option 1: Use synthetic data
make test-pipeline

# Option 2: Use real data
make prepare-data
```

### Issue: `make mlflow-ui` shows no experiments

**Solution:** Run training with MLflow tracking:

```bash
make train-mlflow
```

### Issue: `make mlflow-ui` fails with Alembic migration error

**Error message:**
```
ERROR mlflow.cli: Can't locate revision identified by 'bf29a5ff90ea'
alembic.util.exc.CommandError: Can't locate revision...
```

**Cause:** Database corruption or MLflow version mismatch

**Solution:** Reset the MLflow database:

```bash
make mlflow-reset
make mlflow-ui
```

This will delete the corrupted database and create a fresh one on next start.

### Issue: `make lint` or `make format` not working

**Solution:** Install development dependencies:

```bash
pip install flake8 black pylint mypy
```

### Issue: `make streamlit` or `make gradio` fails

**Solution:** Check if apps exist and dependencies are installed:

```bash
ls app.py app_gradio.py
make setup
```

---

## Environment Variables

Some commands respect environment variables:

### MLflow Tracking URI

```bash
# Override default SQLite backend
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
make train-mlflow
```

### Python Path

```bash
# Add custom Python path
export PYTHONPATH="${PYTHONPATH}:./src"
make train
```

---

## Tips & Best Practices

1. **Always run `make help` first** to see available commands

2. **Use `make full-pipeline` for testing** - it's fast and doesn't require data

3. **Use `make mlflow-ui` to compare experiments** - visual comparison is easier than reading logs

4. **Run `make format` before committing** - maintains code consistency

5. **Use `make view-plots` after training** - quick visual inspection of results

6. **Chain commands with `&&`** for dependent operations:
   ```bash
   make train && make mlflow-ui
   ```

7. **Use `make test-pipeline` for CI/CD** - fast validation without real data

8. **Run `make clean` periodically** - keeps repository tidy

---

## Command Summary Table

| Category | Command | Description | Time |
|----------|---------|-------------|------|
| **Setup** | `make setup` | Install dependencies | ~2 min |
| | `make clean` | Clean caches | <10 sec |
| **Training** | `make train` | Train with real data | 2-5 min |
| | `make train-mlflow` | Train with MLflow | 2-5 min |
| | `make test-pipeline` | Test with synthetic | ~30 sec |
| **MLflow** | `make mlflow-ui` | Start UI | Instant |
| | `make mlflow-info` | Show stats | <1 sec |
| | `make mlflow-export` | Export to CSV | <5 sec |
| **Dashboards** | `make streamlit` | Run Streamlit | ~10 sec |
| | `make gradio` | Run Gradio | ~10 sec |
| **Development** | `make lint` | Quick lint | ~5 sec |
| | `make format` | Format code | ~3 sec |
| **Full** | `make full-pipeline` | Complete (synthetic) | ~2 min |
| | `make full-pipeline-real` | Complete (real) | 5-10 min |

---

## Getting Help

- View all commands: `make help`
- View this guide: `cat docs/MAKEFILE_COMMANDS.md`
- View MLflow guide: `cat docs/MLFLOW_GUIDE.md`
- View implementation guide: `cat docs/README_IMPROVEMENTS.md`

---

**Last Updated:** 2025-11-13  
**Version:** 1.1
