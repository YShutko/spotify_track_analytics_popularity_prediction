"""
Train XGBoost model for Spotify popularity prediction
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """XGBoost model trainer for popularity prediction"""

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_time = 0
        self.metrics = {}

    def load_data(self, data_dir="data/processed"):
        """Load processed training data"""
        data_dir = Path(data_dir)
        logger.info(f"Loading data from {data_dir}")

        self.X_train = pd.read_parquet(data_dir / 'X_train.parquet')
        self.X_test = pd.read_parquet(data_dir / 'X_test.parquet')
        self.y_train = pd.read_parquet(data_dir / 'y_train.parquet').squeeze()
        self.y_test = pd.read_parquet(data_dir / 'y_test.parquet').squeeze()

        logger.info(f"Train: X={self.X_train.shape}, y={self.y_train.shape}")
        logger.info(f"Test: X={self.X_test.shape}, y={self.y_test.shape}")
        return self

    def initialize_model(self):
        """Initialize XGBoost model with hyperparameters"""
        logger.info("Initializing XGBoost model...")

        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric='rmse'
        )

        logger.info("Model initialized")
        return self

    def train(self):
        """Train the model"""
        logger.info("=" * 80)
        logger.info("TRAINING XGBOOST MODEL")
        logger.info("=" * 80)

        start_time = time.time()

        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=50
        )

        self.training_time = time.time() - start_time
        logger.info(f"Training complete in {self.training_time:.2f} seconds")
        return self

    def evaluate(self):
        """Evaluate model performance"""
        logger.info("Evaluating model...")

        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Calculate metrics
        self.metrics = {
            'train': {
                'rmse': float(np.sqrt(mean_squared_error(self.y_train, y_train_pred))),
                'mae': float(mean_absolute_error(self.y_train, y_train_pred)),
                'r2': float(r2_score(self.y_train, y_train_pred))
            },
            'test': {
                'rmse': float(np.sqrt(mean_squared_error(self.y_test, y_test_pred))),
                'mae': float(mean_absolute_error(self.y_test, y_test_pred)),
                'r2': float(r2_score(self.y_test, y_test_pred))
            }
        }

        logger.info(f"Train R²: {self.metrics['train']['r2']:.4f}")
        logger.info(f"Test R²: {self.metrics['test']['r2']:.4f}")
        logger.info(f"Test RMSE: {self.metrics['test']['rmse']:.4f}")
        logger.info(f"Test MAE: {self.metrics['test']['mae']:.4f}")

        return self

    def get_feature_importance(self):
        """Get feature importance"""
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

    def save_model(self, output_dir="outputs/models"):
        """Save model and metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        # Save model (joblib)
        model_path = output_dir / 'xgboost_popularity_model.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Saved: {model_path}")

        # Save model (XGBoost native JSON)
        model_path_json = output_dir / 'xgboost_popularity_model.json'
        self.model.save_model(str(model_path_json))
        logger.info(f"Saved: {model_path_json}")

        # Save feature importance
        feature_importance = self.get_feature_importance()
        feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        logger.info(f"Saved: feature_importance.csv")

        # Save metadata
        metadata = {
            'model_type': 'XGBoost Regressor',
            'target_variable': 'popularity',
            'n_features': self.X_train.shape[1],
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': self.training_time,
            'hyperparameters': {
                'n_estimators': int(self.model.n_estimators),
                'max_depth': int(self.model.max_depth),
                'learning_rate': float(self.model.learning_rate),
                'subsample': float(self.model.subsample),
                'colsample_bytree': float(self.model.colsample_bytree),
                'min_child_weight': int(self.model.min_child_weight),
                'gamma': float(self.model.gamma),
            },
            'performance': self.metrics,
            'top_10_features': feature_importance.head(10)[['feature', 'importance']].to_dict('records')
        }

        metadata_path = output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved: {metadata_path}")

        return output_dir

    def run(self):
        """Run complete training pipeline"""
        self.load_data()
        self.initialize_model()
        self.train()
        self.evaluate()
        output_dir = self.save_model()

        logger.info("=" * 80)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 80)

        return self.model, self.metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    model, metrics = trainer.run()

    # Print summary
    print("\n" + "=" * 80)
    print("XGBOOST TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel Performance:")
    print(f"  Test R²: {metrics['test']['r2']:.4f}")
    print(f"  Test RMSE: {metrics['test']['rmse']:.4f}")
    print(f"  Test MAE: {metrics['test']['mae']:.4f}")

    print(f"\nTop 5 Important Features:")
    feature_importance = trainer.get_feature_importance()
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

    print(f"\nFiles Saved:")
    print(f"  ✓ xgboost_popularity_model.joblib")
    print(f"  ✓ xgboost_popularity_model.json")
    print(f"  ✓ model_metadata.json")
    print(f"  ✓ feature_importance.csv")
