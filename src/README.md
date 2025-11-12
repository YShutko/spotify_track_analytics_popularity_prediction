# Source Code Directory

This directory is intended for Python modules and reusable source code.

## Planned Structure

```
src/
├── data_loader.py       # Data loading and validation utilities
├── preprocessing.py     # Data cleaning and feature engineering
├── features.py          # Feature extraction and transformation
├── visualization.py     # Plotting and chart generation
├── modeling.py          # ML model training and evaluation
└── utils.py            # Helper functions
```

## Usage

Scripts in this directory should be importable from notebooks:

```python
# From notebooks/
import sys
sys.path.append('../src')

from data_loader import load_spotify_data
from preprocessing import clean_data, create_mood_features

# Load and process data
df = load_spotify_data()
df_clean = clean_data(df)
```

## Development Notes

- Keep functions modular and well-documented
- Include type hints for better code clarity
- Write unit tests for utility functions
- Follow PEP 8 style guidelines
