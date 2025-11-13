"""
Hyperparameter Tuning with Optuna for Spotify Track Popularity Prediction

This script uses Optuna to find optimal hyperparameters for the XGBoost model.
Each trial is logged to MLflow (if available) for tracking and comparison.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.trial import Trial

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_utils import adjusted_r2

# Try to import MLflow (fail gracefully if not available)
MLFLOW_AVAILABLE = False
try:
    import mlflow
    from src.mlflow_tracker import MLflowTracker
    MLFLOW_AVAILABLE = True
    print("✅ MLflow available - trials will be logged")
except ImportError:
    print("⚠️  MLflow not available - trials will not be logged")
    print("   Install MLflow with: pip install mlflow")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "cleaned_music_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "tuning"
RANDOM_STATE = 42

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🎯 HYPERPARAMETER TUNING WITH OPTUNA")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


class OptunaXGBoostTuner:
    """Optuna-based hyperparameter tuner for XGBoost with MLflow integration"""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_mlflow: bool = MLFLOW_AVAILABLE,
        experiment_name: str = "spotify_hyperparameter_tuning"
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.use_mlflow = use_mlflow

        # Initialize MLflow if available
        self.mlflow_tracker = None
        if self.use_mlflow:
            try:
                self.mlflow_tracker = MLflowTracker(
                    experiment_name=experiment_name,
                    tracking_uri="sqlite:///mlruns/mlflow.db"
                )
                print(f"✅ MLflow tracking initialized: {experiment_name}")
            except Exception as e:
                print(f"⚠️  MLflow initialization failed: {e}")
                self.use_mlflow = False

        self.best_params = None
        self.best_score = None
        self.study = None

    def define_search_space(self, trial: Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for XGBoost

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters to test
        """
        params = {
            # Tree structure
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),

            # Learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),

            # Sampling
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),

            # Fixed parameters
            'random_state': RANDOM_STATE,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0
        }

        return params

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization

        Args:
            trial: Optuna trial object

        Returns:
            Validation RMSE (to minimize)
        """
        # Get hyperparameters for this trial
        params = self.define_search_space(trial)

        # Start MLflow run for this trial if available
        mlflow_run = None
        if self.use_mlflow and self.mlflow_tracker:
            try:
                mlflow_run = self.mlflow_tracker.start_run(
                    run_name=f"optuna_trial_{trial.number}",
                    tags={
                        'framework': 'optuna',
                        'trial_number': str(trial.number),
                        'optimization': 'hyperparameter_tuning'
                    }
                )
            except Exception as e:
                print(f"⚠️  Failed to start MLflow run: {e}")

        try:
            # Train model
            model = XGBRegressor(**params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )

            # Evaluate on validation set
            y_val_pred = model.predict(self.X_val)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, y_val_pred))
            val_mae = mean_absolute_error(self.y_val, y_val_pred)
            val_r2 = r2_score(self.y_val, y_val_pred)
            val_adj_r2 = adjusted_r2(val_r2, len(self.y_val), self.X_val.shape[1])

            # Log to MLflow if available
            if self.use_mlflow and self.mlflow_tracker and mlflow_run:
                try:
                    # Log parameters
                    self.mlflow_tracker.log_params(params)

                    # Log metrics
                    self.mlflow_tracker.log_metrics({
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'val_r2': val_r2,
                        'val_adjusted_r2': val_adj_r2,
                        'trial_number': trial.number
                    })

                    # End run
                    self.mlflow_tracker.end_run(status="FINISHED")
                except Exception as e:
                    print(f"⚠️  Failed to log to MLflow: {e}")

            # Report intermediate value for pruning
            trial.report(val_rmse, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return val_rmse

        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            if self.use_mlflow and self.mlflow_tracker and mlflow_run:
                try:
                    self.mlflow_tracker.end_run(status="FAILED")
                except:
                    pass
            raise

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None for no timeout)
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            Dictionary with best parameters and results
        """
        print(f"\n{'='*80}")
        print(f"🚀 STARTING OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Trials: {n_trials}")
        print(f"Timeout: {timeout if timeout else 'None'}")
        print(f"Parallel jobs: {n_jobs}")
        print(f"MLflow logging: {'Enabled' if self.use_mlflow else 'Disabled'}")
        print()

        # Create Optuna study
        self.study = optuna.create_study(
            direction='minimize',  # Minimize RMSE
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\n{'='*80}")
        print("✅ OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Best validation RMSE: {self.best_score:.4f}")
        print(f"Best trial number: {self.study.best_trial.number}")
        print(f"\n📊 Best Hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        # Save results
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_number': self.study.best_trial.number,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat(),
            'random_state': RANDOM_STATE
        }

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_results_{timestamp}.json"

        output_path = OUTPUT_DIR / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

        # Also save best params in the format used by load_config
        best_params_path = BASE_DIR / "config" / "xgboost_params_tuned.json"
        tuned_params = {
            **results['best_params'],
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
            'eval_metric': 'rmse'
        }

        with open(best_params_path, 'w') as f:
            json.dump(tuned_params, f, indent=2)

        print(f"💾 Best params saved to: {best_params_path}")

        return output_path


def load_data(use_synthetic: bool = False) -> tuple:
    """Load and prepare data for tuning"""
    print("\n📂 Loading Data")
    print("-" * 80)

    if use_synthetic or not DATA_PATH.exists():
        if not use_synthetic:
            print(f"⚠️  Data file not found: {DATA_PATH}")
            print("   Generating synthetic data for tuning...")

        # Generate synthetic data
        np.random.seed(RANDOM_STATE)
        n_samples = 1000

        data = {
            'danceability': np.random.randn(n_samples),
            'energy': np.random.randn(n_samples),
            'loudness': np.random.randn(n_samples),
            'acousticness': np.random.randn(n_samples),
            'tempo': np.random.randn(n_samples),
            'valence': np.random.randn(n_samples),
            'instrumentalness': np.random.randn(n_samples),
        }

        popularity = (
            10 * data['energy'] +
            8 * data['danceability'] +
            5 * data['valence'] +
            np.random.randn(n_samples) * 5 +
            50
        )
        popularity = np.clip(popularity, 0, 100)

        df = pd.DataFrame(data)
        df['popularity'] = popularity

        print(f"✅ Generated synthetic data: {len(df):,} samples")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Loaded real data: {len(df):,} samples")

    # Prepare features and target
    audio_features = [
        'danceability', 'energy', 'loudness', 'acousticness',
        'tempo', 'valence', 'instrumentalness'
    ]

    # Add additional features if available
    additional_features = ['duration_min', 'release_year', 'speechiness', 'liveness']
    audio_features += [f for f in additional_features if f in df.columns]

    X = df[audio_features].copy()
    y = df['popularity'].copy()

    # Remove any rows with NaN or infinite values
    mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1) | y.isnull() | np.isinf(y))
    X = X[mask]
    y = y[mask]

    print(f"✅ Clean dataset: {len(X):,} samples, {len(audio_features)} features")
    print(f"   Features: {audio_features}")

    # Split data: 64% train, 16% validation, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main(
    n_trials: int = 50,
    timeout: Optional[int] = None,
    use_synthetic: bool = False,
    n_jobs: int = 1
):
    """Main execution function"""

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(use_synthetic=use_synthetic)

    # Create tuner
    tuner = OptunaXGBoostTuner(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        use_mlflow=MLFLOW_AVAILABLE
    )

    # Run optimization
    results = tuner.optimize(
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs
    )

    # Save results
    tuner.save_results(results)

    # Evaluate best model on test set
    print(f"\n{'='*80}")
    print("📊 EVALUATING BEST MODEL ON TEST SET")
    print(f"{'='*80}")

    best_model = XGBRegressor(**tuner.best_params)
    best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test.shape[1])

    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  Adj R²: {test_adj_r2:.4f}")

    print(f"\n{'='*80}")
    print("✨ HYPERPARAMETER TUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n🎯 Next steps:")
    print(f"  - Review results in: {OUTPUT_DIR}")
    print(f"  - Best params saved to: config/xgboost_params_tuned.json")
    if MLFLOW_AVAILABLE:
        print(f"  - View trials in MLflow UI: make mlflow-ui")
    print(f"  - Use tuned params: Update config/xgboost_params.json")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: None)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real data"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, -1 for all cores)"
    )

    args = parser.parse_args()

    main(
        n_trials=args.trials,
        timeout=args.timeout,
        use_synthetic=args.synthetic,
        n_jobs=args.jobs
    )
