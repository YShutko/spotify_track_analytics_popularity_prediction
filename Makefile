.PHONY: help install dashboard pipeline etl fe train test lint format clean all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
STREAMLIT := streamlit
VENV := .venv
SRC_DIR := src
NOTEBOOKS_DIR := notebooks
DATA_DIR := data

help: ## Show this help message
	@echo "Spotify Track Analytics - Available Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 pylint nbdime papermill ipykernel
	$(PYTHON) -m ipykernel install --user --name spotify_env
	nbdime config-git --enable --global
	@echo "✅ Installation complete!"

dashboard: ## Launch Streamlit dashboard
	@echo "🚀 Launching Streamlit dashboard..."
	@echo "Dashboard will open at http://localhost:8501"
	@echo "Press Ctrl+C to stop"
	$(STREAMLIT) run app.py

pipeline: ## Run complete ETL -> FE -> ML pipeline
	@echo "🔄 Running complete pipeline..."
	$(PYTHON) run_pipeline.py

etl: ## Run ETL pipeline only
	@echo "📊 Running ETL pipeline..."
	$(PYTHON) $(SRC_DIR)/etl_pipeline.py

fe: ## Run Feature Engineering only
	@echo "🔧 Running Feature Engineering..."
	$(PYTHON) $(SRC_DIR)/feature_engineering.py

train: ## Train ML model only
	@echo "🤖 Training ML model..."
	$(PYTHON) $(SRC_DIR)/train_model.py

test: ## Run pipeline verification tests
	@echo "🧪 Running tests..."
	$(PYTHON) $(SRC_DIR)/test_pipeline.py

lint: ## Run linting checks (flake8)
	@echo "🔍 Running flake8 linting..."
	flake8 $(SRC_DIR) app.py run_pipeline.py --max-line-length=120 --extend-ignore=E203,W503

lint-strict: ## Run strict linting (pylint)
	@echo "🔍 Running pylint (strict)..."
	pylint $(SRC_DIR)/*.py app.py run_pipeline.py --max-line-length=120

format: ## Format code with black
	@echo "✨ Formatting code with black..."
	black $(SRC_DIR) app.py run_pipeline.py --line-length=120
	@echo "✅ Code formatted!"

format-check: ## Check if code is formatted
	@echo "🔍 Checking code formatting..."
	black $(SRC_DIR) app.py run_pipeline.py --check --line-length=120

nbconvert: ## Convert notebooks to Python scripts
	@echo "📓 Converting notebooks to scripts..."
	jupyter nbconvert --to script $(NOTEBOOKS_DIR)/*.ipynb
	@echo "✅ Notebooks converted!"

nbdiff: ## Show notebook diffs with nbdime
	@echo "📊 Opening nbdime diff tool..."
	nbdiff-web

clean: ## Clean generated files
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__ $(SRC_DIR)/__pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc $(SRC_DIR)/*.pyc
	rm -rf $(NOTEBOOKS_DIR)/*_output.ipynb
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

clean-data: ## Remove processed data (keeps raw data)
	@echo "⚠️  Warning: This will delete processed data files"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR)/processed/*; \
		echo "✅ Processed data removed"; \
	else \
		echo "❌ Cancelled"; \
	fi

clean-models: ## Remove trained models
	@echo "⚠️  Warning: This will delete trained models"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf outputs/models/*; \
		echo "✅ Models removed"; \
	else \
		echo "❌ Cancelled"; \
	fi

requirements: ## Generate/update requirements.txt
	@echo "📋 Updating requirements.txt..."
	$(PIP) freeze > requirements_frozen.txt
	@echo "✅ Requirements saved to requirements_frozen.txt"

dev: ## Install development dependencies
	@echo "🛠️  Installing development tools..."
	$(PIP) install -U black flake8 pylint pytest nbdime papermill ipykernel jupyter
	@echo "✅ Dev tools installed!"

status: ## Show pipeline status
	@echo "📊 Pipeline Status"
	@echo "=================="
	@echo ""
	@echo "Data Files:"
	@ls -lh $(DATA_DIR)/raw/dataset.csv 2>/dev/null && echo "  ✅ Raw data present" || echo "  ❌ Raw data missing"
	@ls -lh $(DATA_DIR)/processed/cleaned_spotify_data.parquet 2>/dev/null && echo "  ✅ Cleaned data present" || echo "  ❌ Cleaned data missing"
	@ls -lh $(DATA_DIR)/processed/X_train.parquet 2>/dev/null && echo "  ✅ Training data present" || echo "  ❌ Training data missing"
	@echo ""
	@echo "Models:"
	@ls -lh outputs/models/xgboost_popularity_model.joblib 2>/dev/null && echo "  ✅ Model trained" || echo "  ❌ Model not trained"
	@echo ""
	@echo "App:"
	@ls -lh app.py 2>/dev/null && echo "  ✅ Dashboard ready" || echo "  ❌ Dashboard missing"

all: clean pipeline test ## Run full workflow: clean -> pipeline -> test
	@echo "✅ Full workflow complete!"

quick-start: install pipeline dashboard ## Quick start: install -> pipeline -> dashboard
	@echo "🎉 Quick start complete!"

# Notebook execution targets
nb-etl: ## Run ETL notebook with papermill
	@echo "📓 Running ETL notebook..."
	papermill $(NOTEBOOKS_DIR)/00_ETL_Pipeline.ipynb $(NOTEBOOKS_DIR)/00_ETL_Pipeline_output.ipynb

nb-fe: ## Run Feature Engineering notebook with papermill
	@echo "📓 Running Feature Engineering notebook..."
	papermill $(NOTEBOOKS_DIR)/02_Feature_Engineering.ipynb $(NOTEBOOKS_DIR)/02_Feature_Engineering_output.ipynb

nb-ml: ## Run ML notebook with papermill
	@echo "📓 Running ML training notebook..."
	papermill $(NOTEBOOKS_DIR)/03_ML_XGBoost_Model.ipynb $(NOTEBOOKS_DIR)/03_ML_XGBoost_Model_output.ipynb

nb-all: nb-etl nb-fe nb-ml ## Run all notebooks with papermill
	@echo "✅ All notebooks executed!"
