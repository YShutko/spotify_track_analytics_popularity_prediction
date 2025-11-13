# ML Pipeline Improvements - Quick Start Guide

This document provides a quick overview of the ML pipeline improvements and how to use them.

## 🚀 Quick Start

### Option 1: Run the Improved Notebook (Recommended)

```bash
jupyter notebook notebooks/04_Improved_ML_Pipeline.ipynb
```

This notebook includes:
- All Phase 1 & Phase 2 improvements
- Interactive SHAP visualizations
- Step-by-step execution with explanations
- Works with existing processed data in `data/processed/`

### Option 2: Run the Python Script

```bash
source .venv/bin/activate
python src/improved_ml_pipeline.py
```

**Note**: Requires `cleaned_music_data.csv` in the root directory.

## 📁 Project Structure

```
├── config/
│   └── xgboost_params.json          # Model hyperparameters
├── notebooks/
│   ├── 03_ML_XGBoost_Model.ipynb    # Original model (baseline)
│   └── 04_Improved_ML_Pipeline.ipynb # ⭐ NEW: Improved pipeline
├── src/
│   ├── ml_utils.py                   # Utility functions
│   ├── improved_ml_pipeline.py       # Standalone script
│   └── test_pipeline.py              # Synthetic data test
├── outputs/
│   ├── plots/                        # Visualizations
│   ├── models/                       # Trained models
│   └── metadata/                     # Model metadata
└── dev_docs/
    ├── ML_PIPELINE_IMPROVEMENTS_SPEC.md # Full specification
    └── reference_images/             # Original suggestions
```

## ✨ What's New

### Phase 1 (Critical - HIGH Priority)

| Feature | Benefit | Location |
|---------|---------|----------|
| **Data Validation** | Prevents silent failures | `ml_utils.py:validate_train_test_features()` |
| **Separate Validation Set** | Eliminates data leakage | 64% train / 16% val / 20% test split |
| **SHAP Values** ⭐ | Model interpretability | 2 plots + CSV export |
| **Git Tracking** | Reproducibility | Commit hash in metadata |
| **Environment Metadata** | Version control | Python, libraries, platform |

### Phase 2 (Enhanced - MEDIUM Priority)

| Feature | Benefit | Location |
|---------|---------|----------|
| **JSON Config** | Easy parameter management | `config/xgboost_params.json` |
| **Learning Curves** | Overfitting detection | Train/val RMSE plot |
| **Adjusted R²** | Better model comparison | Accounts for # of features |
| **Correlation Heatmap** | Feature relationships | EDA extension |
| **QQ Plots** | Residual normality check | Statistical validation |

## 📊 Generated Outputs

### Visualizations (10+ plots in `outputs/plots/`)

1. **Learning curves** - Detect overfitting
2. **Actual vs predicted** (train/val/test) - Model accuracy
3. **Prediction density** - For large datasets
4. **Residual analysis** - Error patterns
5. **QQ plot** - Normality check
6. **Correlation heatmap** - Feature-target relationships
7. **Feature importance** (standard) - XGBoost gain
8. **SHAP summary bar** ⭐ - Global importance
9. **SHAP beeswarm** ⭐ - Feature impact distribution
10. **SHAP waterfall** ⭐ - Individual predictions

### Model Artifacts

- **Model file**: `xgb_model_YYYYMMDD_HHMMSS.joblib`
- **Metadata JSON**: Complete tracking info (git, env, metrics, features)
- **SHAP export CSV**: Per-sample SHAP values for dashboard integration

## 🎯 Key Metrics Tracked

```python
{
  "test_r2": 0.XXXX,
  "test_adjusted_r2": 0.XXXX,  # NEW: Adjusted for # of features
  "test_rmse": X.XXXX,
  "test_mae": X.XXXX,
  "git_commit": "abc123...",     # NEW: Reproducibility
  "environment": {                # NEW: Version tracking
    "python_version": "3.12.x",
    "xgboost_version": "2.x.x",
    "shap_version": "0.50.x"
  }
}
```

## 🔬 SHAP Values Explained

**Why SHAP?**
- More reliable than standard feature importance
- Shows feature impact direction (positive/negative)
- Explains individual predictions
- Builds stakeholder trust in model decisions

**How to interpret SHAP plots:**

### Summary Bar Plot
- Shows average absolute impact of each feature
- Higher = more important globally

### Beeswarm Plot
- Each dot = one sample
- X-axis = Impact on prediction
- Color = Feature value (red=high, blue=low)
- Example: High values of "energy" (red dots) on the right side means high energy increases popularity

### Waterfall Plot
- Explains a single prediction
- Start with base value (average prediction)
- Each bar shows how a feature pushes prediction up (red) or down (blue)
- End with final prediction

## 📈 Before vs After Comparison

| Metric | Original Notebook | Improved Pipeline | Improvement |
|--------|-------------------|-------------------|-------------|
| Validation Set | ❌ Reused test set | ✅ Separate 16% split | No data leakage |
| Feature Importance | ✅ XGBoost gain | ✅ SHAP + XGBoost | Better interpretability |
| Overfitting Detection | ❌ Manual check | ✅ Learning curves | Visual diagnosis |
| Reproducibility | ❌ None | ✅ Git + environment | Full tracking |
| Model Comparison | ✅ R² only | ✅ R² + Adjusted R² | Fair comparison |
| Residual Analysis | ✅ Scatter + histogram | ✅ + QQ plot | Statistical validation |
| Config Management | ❌ Hardcoded | ✅ JSON file | Version controlled |

## 🛠️ Customization

### Adjust Hyperparameters

Edit `config/xgboost_params.json`:

```json
{
  "max_depth": 8,           // Increase for more complex patterns
  "learning_rate": 0.05,    // Decrease for better generalization
  "n_estimators": 300       // Increase for more iterations
}
```

### Change Data Splits

In notebook or script, modify:

```python
# Change validation size (default: 20% of training data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,  # Change this
    random_state=42
)
```

## 🐛 Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Run from project root or add to Python path:
```python
import sys
sys.path.append('/path/to/project/root')
```

### Issue: SHAP computation is slow

**Solution**: Pipeline auto-samples 10K points. Adjust in code:
```python
X_test_shap = X_test.sample(n=5000, random_state=42)  # Reduce sample size
```

### Issue: "Git commit is None"

**Solution**: This is normal if not in a git repo. Pipeline continues normally.

## 📚 Documentation

- **Full Specification**: `dev_docs/ML_PIPELINE_IMPROVEMENTS_SPEC.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Original Suggestions**: `dev_docs/reference_images/` (8 images)
- **Source Code**: `src/improved_ml_pipeline.py`
- **Utilities**: `src/ml_utils.py`

## 🎓 Learning Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Model Interpretability Guide](https://christophm.github.io/interpretable-ml-book/)

## 🚦 Next Steps

1. **Run the improved notebook**: `jupyter notebook notebooks/04_Improved_ML_Pipeline.ipynb`
2. **Compare results**: Check if adjusted R² differs significantly from R²
3. **Analyze SHAP values**: Identify which features drive popularity
4. **Experiment with hyperparameters**: Edit `config/xgboost_params.json`
5. **Integrate into dashboard**: Use SHAP CSV for track-level explanations

## 💡 Pro Tips

1. **Use the notebook for exploration** - Interactive SHAP plots are better in Jupyter
2. **Use the script for production** - Automated pipeline for batch processing
3. **Check learning curves** - If validation RMSE diverges from training, you're overfitting
4. **Compare SHAP vs standard importance** - They often differ! SHAP is more reliable
5. **Explain predictions to stakeholders** - Use SHAP waterfall plots

## 🤝 Contributing

To add more improvements:

1. Update `config/xgboost_params.json` with new parameters
2. Add utility functions to `src/ml_utils.py`
3. Document changes in this README
4. Commit with descriptive message

## 📊 Expected Performance

With the improved pipeline on Spotify dataset (~114K tracks):

- **Test R²**: 0.78 - 0.85 (expected range)
- **Test Adjusted R²**: ~0.02 lower than R² (normal)
- **Training time**: 2-5 minutes (depending on hardware)
- **SHAP computation**: 3-10 minutes (for 10K samples)

---

**Status**: ✅ Production Ready

**Last Updated**: 2025-11-13

**Questions?** See `dev_docs/ML_PIPELINE_IMPROVEMENTS_SPEC.md` for detailed documentation.
