"""
Test end-to-end ML pipeline
Verifies all components work together correctly
"""

import pandas as pd
import joblib
import json
from pathlib import Path
import sys

def test_data_files():
    """Test that all data files exist and are readable"""
    print("=" * 80)
    print("TESTING DATA FILES")
    print("=" * 80)

    files_to_check = [
        'data/raw/dataset.csv',
        'data/processed/cleaned_spotify_data.parquet',
        'data/processed/X_train.parquet',
        'data/processed/X_test.parquet',
        'data/processed/y_train.parquet',
        'data/processed/y_test.parquet',
        'data/processed/ml_ready_data.parquet',
    ]

    all_exist = True
    for file_path in files_to_check:
        path = Path(file_path)
        exists = path.exists()
        size = path.stat().st_size / 1024**2 if exists else 0
        status = "✅" if exists else "❌"
        print(f"{status} {file_path:50s} {size:8.2f} MB")
        if not exists:
            all_exist = False

    return all_exist

def test_model_files():
    """Test that model files exist and are loadable"""
    print("\n" + "=" * 80)
    print("TESTING MODEL FILES")
    print("=" * 80)

    model_files = [
        'outputs/models/xgboost_popularity_model.joblib',
        'outputs/models/xgboost_popularity_model.json',
        'outputs/models/model_metadata.json',
        'outputs/models/feature_importance.csv',
    ]

    all_exist = True
    for file_path in model_files:
        path = Path(file_path)
        exists = path.exists()
        size = path.stat().st_size / 1024 if exists else 0
        status = "✅" if exists else "❌"
        print(f"{status} {file_path:50s} {size:8.2f} KB")
        if not exists:
            all_exist = False

    return all_exist

def test_model_loading():
    """Test that model can be loaded and used"""
    print("\n" + "=" * 80)
    print("TESTING MODEL LOADING")
    print("=" * 80)

    try:
        # Load model
        model = joblib.load('outputs/models/xgboost_popularity_model.joblib')
        print("✅ Model loaded successfully")

        # Load test data
        X_test = pd.read_parquet('data/processed/X_test.parquet')
        y_test = pd.read_parquet('data/processed/y_test.parquet').squeeze()
        print(f"✅ Test data loaded: X={X_test.shape}, y={y_test.shape}")

        # Make predictions
        predictions = model.predict(X_test[:100])
        print(f"✅ Predictions generated: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:5]}")
        print(f"   Actual values: {y_test[:5].values}")

        # Load metadata
        with open('outputs/models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded")
        print(f"   Model type: {metadata['model_type']}")
        print(f"   Test R²: {metadata['performance']['test']['r2']:.4f}")
        print(f"   Test RMSE: {metadata['performance']['test']['rmse']:.4f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pipeline_scripts():
    """Test that pipeline scripts exist"""
    print("\n" + "=" * 80)
    print("TESTING PIPELINE SCRIPTS")
    print("=" * 80)

    scripts = [
        'src/etl_pipeline.py',
        'src/feature_engineering.py',
        'src/train_model.py',
    ]

    all_exist = True
    for script_path in scripts:
        path = Path(script_path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {script_path}")
        if not exists:
            all_exist = False

    return all_exist

def test_notebooks():
    """Test that notebooks exist"""
    print("\n" + "=" * 80)
    print("TESTING NOTEBOOKS")
    print("=" * 80)

    notebooks = [
        'notebooks/01_ETL_Validation.ipynb',
        'notebooks/02_Feature_Engineering.ipynb',
        'notebooks/03_ML_XGBoost_Model.ipynb',
    ]

    all_exist = True
    for notebook_path in notebooks:
        path = Path(notebook_path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {notebook_path}")
        if not exists:
            all_exist = False

    return all_exist

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ML PIPELINE VERIFICATION TEST" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = {
        'Data Files': test_data_files(),
        'Model Files': test_model_files(),
        'Model Loading': test_model_loading(),
        'Pipeline Scripts': test_pipeline_scripts(),
        'Notebooks': test_notebooks(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {test_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - PIPELINE IS READY!")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK ERRORS ABOVE")
    print("=" * 80)
    print()

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
