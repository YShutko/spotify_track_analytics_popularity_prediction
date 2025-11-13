.PHONY: help setup clean test mlflow-ui mlflow-clean train notebook lint format streamlit gradio

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

setup: ## Install dependencies and setup environment
	@echo "📦 Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Setup complete!"

clean: ## Clean generated files and caches
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".DS_Store" -delete
	@echo "✅ Cleanup complete!"

##@ MLflow

mlflow-ui: ## Start MLflow UI with SQLite backend
	@echo "🚀 Starting MLflow UI..."
	@echo "   Backend: SQLite (mlruns/mlflow.db)"
	@echo "   URL: http://127.0.0.1:5000"
	@echo ""
	@mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

mlflow-clean: ## Clean all MLflow runs and artifacts
	@echo "⚠️  This will delete all MLflow experiments, runs, and artifacts!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf mlruns mlartifacts; \
		echo "✅ MLflow data deleted!"; \
	else \
		echo "❌ Cancelled"; \
	fi

mlflow-reset: ## Reset MLflow database (fixes inconsistent state)
	@echo "🔧 Resetting MLflow database..."
	@echo "   This will delete the database but preserve experiments and artifacts"
	@pkill -f "mlflow ui" 2>/dev/null || true
	@sleep 1
	@rm -f mlruns/.trash/* 2>/dev/null || true
	@if [ -f mlruns/mlflow.db ]; then \
		rm -f mlruns/mlflow.db mlruns/mlflow.db-shm mlruns/mlflow.db-wal; \
		echo "✅ Deleted corrupted database files"; \
	else \
		echo "📝 No database file found"; \
	fi
	@mkdir -p mlruns mlartifacts
	@echo "✅ MLflow reset complete!"
	@echo "   Database will be recreated on next start"
	@echo "   Run: make mlflow-ui"

mlflow-export: ## Export MLflow experiments to CSV
	@echo "📊 Exporting MLflow experiments..."
	@mkdir -p exports
	@python -c "import mlflow; import pandas as pd; \
		mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); \
		runs = mlflow.search_runs(); \
		runs.to_csv('exports/mlflow_runs.csv', index=False); \
		print(f'✅ Exported {len(runs)} runs to exports/mlflow_runs.csv')"

mlflow-info: ## Show MLflow tracking information
	@echo "📊 MLflow Information:"
	@echo "   Tracking URI: sqlite:///mlruns/mlflow.db"
	@echo "   Artifacts: ./mlartifacts"
	@echo ""
	@python -c "import mlflow; \
		mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); \
		experiments = mlflow.search_experiments(); \
		print(f'   Total Experiments: {len(experiments)}'); \
		for exp in experiments: \
			runs = mlflow.search_runs([exp.experiment_id]); \
			print(f'   - {exp.name}: {len(runs)} runs');" 2>/dev/null || echo "   No experiments found"
	@echo ""
	@ps aux | grep "mlflow ui" | grep -v grep | head -1 | awk '{print "   MLflow UI Status: Running (PID " $$2 ")"}' || echo "   MLflow UI Status: Not running"

mlflow-stop: ## Stop MLflow UI server
	@echo "🛑 Stopping MLflow UI..."
	@pkill -f "mlflow ui" 2>/dev/null || true
	@pkill -f "mlflow.server:app" 2>/dev/null || true
	@sleep 1
	@if ps aux | grep -E "(mlflow ui|mlflow.server)" | grep -v grep > /dev/null; then \
		echo "⚠️  Some MLflow processes may still be running"; \
	else \
		echo "✅ MLflow UI stopped"; \
	fi

##@ Training

train: ## Run improved ML pipeline
	@echo "🤖 Training model with improved pipeline..."
	@python src/improved_ml_pipeline.py

train-mlflow: ## Run improved ML pipeline with MLflow tracking (real data)
	@echo "🤖 Training model with MLflow tracking..."
	@python src/improved_ml_pipeline_mlflow.py

test-pipeline: ## Test pipeline with synthetic data
	@echo "🧪 Testing pipeline with synthetic data..."
	@python src/test_pipeline.py
	@python src/improved_ml_pipeline.py
	@echo "✅ Pipeline test complete!"

##@ Notebooks

notebook: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook..."
	@jupyter notebook

notebook-improved: ## Open improved ML pipeline notebook
	@echo "📓 Opening improved ML pipeline notebook..."
	@jupyter notebook notebooks/04_Improved_ML_Pipeline.ipynb

##@ Dashboards

streamlit: ## Run Streamlit dashboard
	@echo "🎨 Starting Streamlit dashboard..."
	@echo "   URL: http://localhost:8501"
	@streamlit run app.py

gradio: ## Run Gradio dashboard
	@echo "🎨 Starting Gradio dashboard..."
	@echo "   URL: http://localhost:7860"
	@python app_gradio.py

hf-space: ## Run Hugging Face Space app locally
	@echo "🤗 Starting Hugging Face Space app..."
	@cd hf_space && streamlit run app.py

##@ Development

lint: ## Run code quality checks
	@echo "🔍 Running linting checks..."
	@python -m flake8 src/ --max-line-length=100 --ignore=E203,W503 || echo "⚠️  Install flake8: pip install flake8"

lint-all: ## Run comprehensive linting (flake8, pylint, mypy)
	@echo "🔍 Running comprehensive linting..."
	@echo "Running flake8..."
	@python -m flake8 src/ app.py app_gradio.py --max-line-length=100 --ignore=E203,W503 || echo "⚠️  Install flake8: pip install flake8"
	@echo "Running pylint..."
	@python -m pylint src/ --max-line-length=100 --disable=C0103,C0114,C0115,C0116 || echo "⚠️  Install pylint: pip install pylint"
	@echo "Running mypy..."
	@python -m mypy src/ --ignore-missing-imports || echo "⚠️  Install mypy: pip install mypy"

format: ## Format code with black
	@echo "✨ Formatting code..."
	@python -m black src/ --line-length=100 || echo "⚠️  Install black: pip install black"

format-all: ## Format all Python files (src + apps)
	@echo "✨ Formatting all Python files..."
	@python -m black src/ app.py app_gradio.py --line-length=100 || echo "⚠️  Install black: pip install black"

##@ Git

git-status: ## Show git status with useful information
	@echo "📊 Git Status:"
	@git status -sb
	@echo ""
	@echo "Recent commits:"
	@git log --oneline -5

git-push: ## Push commits to remote
	@echo "🚀 Pushing to remote..."
	@git push origin main

##@ Outputs

view-plots: ## Open outputs directory to view plots
	@echo "📊 Opening outputs/plots directory..."
	@open outputs/plots/ 2>/dev/null || xdg-open outputs/plots/ 2>/dev/null || echo "Plots location: outputs/plots/"

view-models: ## Show saved models
	@echo "💾 Saved Models:"
	@ls -lh outputs/models/*.joblib 2>/dev/null || echo "No models found"

view-metadata: ## Show model metadata
	@echo "📋 Model Metadata:"
	@ls -lh outputs/metadata/*.json 2>/dev/null || echo "No metadata found"
	@echo ""
	@echo "Latest metadata:"
	@ls -t outputs/metadata/*.json 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No metadata found"

##@ Documentation

docs: ## Show project documentation
	@echo "📚 Project Documentation:"
	@echo "   - Full Spec: dev_docs/ML_PIPELINE_IMPROVEMENTS_SPEC.md"
	@echo "   - Implementation Summary: IMPLEMENTATION_SUMMARY.md"
	@echo "   - Quick Start: README_IMPROVEMENTS.md"
	@echo "   - Source Code: src/README.md"

view-spec: ## View ML pipeline improvements specification
	@cat dev_docs/ML_PIPELINE_IMPROVEMENTS_SPEC.md | less

##@ Docker (Optional)

docker-build: ## Build Docker image for the project
	@echo "🐳 Building Docker image..."
	@docker build -t spotify-ml-pipeline .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	@docker run -p 5000:5000 -p 8888:8888 -v $(PWD):/app spotify-ml-pipeline

##@ All-in-one

full-pipeline: clean setup test-pipeline ## Run full pipeline with synthetic data
	@echo "✅ Full pipeline complete!"
	@echo ""
	@echo "To view results:"
	@echo "  - Plots: make view-plots"
	@echo "  - MLflow UI: make mlflow-ui"

full-pipeline-real: clean setup prepare-data train ## Run full pipeline with real data
	@echo "✅ Full pipeline complete!"
	@echo ""
	@echo "To view results:"
	@echo "  - Plots: make view-plots"
	@echo "  - MLflow UI: make mlflow-ui"

prepare-data: ## Prepare cleaned data from processed parquet files
	@echo "📊 Preparing cleaned data from parquet files..."
	@python -c "import pandas as pd; \
		import os; \
		if not os.path.exists('data/processed/ml_ready_data.parquet'): \
			print('❌ Error: data/processed/ml_ready_data.parquet not found'); \
			print('   Please run ETL pipeline first or use: make test-pipeline'); \
			exit(1); \
		df = pd.read_parquet('data/processed/ml_ready_data.parquet'); \
		df.to_csv('cleaned_music_data.csv', index=False); \
		print(f'✅ Created cleaned_music_data.csv with {len(df):,} rows')"

