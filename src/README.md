# Improved ML Pipeline - Implementation Guide

This directory contains the improved ML pipeline implementation with all Phase 1 and Phase 2 enhancements from the specification document.

## Files

- **`ml_utils.py`**: Utility functions for data validation, metrics, and metadata tracking
- **`improved_ml_pipeline.py`**: Complete ML pipeline with all improvements
- **`__init__.py`**: Package initialization

## Prerequisites

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Cleaned Data

Before running the improved pipeline, you need to generate the cleaned data:

```bash
# Run the EDA notebook to generate cleaned_music_data.csv
jupyter notebook notebooks/Hackathon2Music.ipynb
```

The EDA notebook will:
- Extract data from `data/dataset.csv.zip`
- Clean and preprocess the data
- Save `cleaned_music_data.csv` in the project root

## Running the Improved Pipeline

Once the cleaned data is available:

```bash
source .venv/bin/activate
python src/improved_ml_pipeline.py
```

## What Gets Generated

The pipeline will create the following outputs in the `outputs/` directory.

See the full specification document for details: `dev_docs/ML_PIPELINE_IMPROVEMENTS_SPEC.md`

## Improvements Implemented

### Phase 1 (Critical)
- Data validation
- Separate validation set
- SHAP values (CRITICAL)
- Git commit tracking
- Environment metadata

### Phase 2 (Enhanced)
- JSON configuration
- Learning curves
- Adjusted R²
- Correlation heatmap
- QQ plots

