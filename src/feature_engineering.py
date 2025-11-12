"""
Feature Engineering Pipeline for ML
Prepares cleaned data for machine learning modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for Spotify ML"""

    def __init__(self, input_path: str = "data/processed/cleaned_spotify_data.parquet"):
        self.input_path = Path(input_path)
        self.df = None
        self.scaler = StandardScaler()
        self.le_genre = LabelEncoder()
        self.target_variable = 'popularity'

    def load_data(self):
        """Load cleaned dataset"""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        return self

    def create_interaction_features(self):
        """Create interaction and polynomial features"""
        logger.info("Creating interaction features...")

        # Feature interactions
        self.df['energy_danceability'] = self.df['energy'] * self.df['danceability']
        self.df['valence_energy'] = self.df['valence'] * self.df['energy']
        self.df['acousticness_energy'] = self.df['acousticness'] * self.df['energy']

        # Polynomial features
        self.df['energy_squared'] = self.df['energy'] ** 2
        self.df['danceability_squared'] = self.df['danceability'] ** 2
        self.df['valence_squared'] = self.df['valence'] ** 2

        # Duration categories
        self.df['is_short_track'] = (self.df['duration_min'] < 3).astype(int)
        self.df['is_long_track'] = (self.df['duration_min'] > 5).astype(int)

        # Mood indicators
        self.df['high_energy_happy'] = ((self.df['energy'] > 0.7) & (self.df['valence'] > 0.7)).astype(int)
        self.df['low_energy_sad'] = ((self.df['energy'] < 0.3) & (self.df['valence'] < 0.3)).astype(int)

        logger.info("Created 10 interaction features")
        return self

    def encode_features(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical features...")

        # Binary encoding
        self.df['explicit'] = self.df['explicit'].astype(int)

        # One-hot encode low-cardinality categoricals
        low_cardinality = ['mode', 'time_signature', 'mood_energy', 'energy_category', 'tempo_category']

        for col in low_cardinality:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)

        # Label encode track_genre
        if 'track_genre' in self.df.columns:
            self.df['track_genre_encoded'] = self.le_genre.fit_transform(self.df['track_genre'])
            self.df.drop('track_genre', axis=1, inplace=True)
            logger.info(f"Encoded {len(self.le_genre.classes_)} unique genres")

        logger.info(f"Total features after encoding: {len(self.df.columns)}")
        return self

    def scale_features(self):
        """Scale numeric features"""
        logger.info("Scaling numeric features...")

        features_to_scale = ['duration_ms', 'loudness', 'tempo', 'duration_min', 'track_genre_encoded', 'key']
        features_to_scale = [f for f in features_to_scale if f in self.df.columns]

        self.df[features_to_scale] = self.scaler.fit_transform(self.df[features_to_scale])
        logger.info(f"Scaled {len(features_to_scale)} features")
        return self

    def prepare_for_ml(self):
        """Prepare final dataset for ML"""
        logger.info("Preparing data for ML...")

        # Features to exclude
        exclude_features = [
            'Unnamed: 0', 'track_id', 'track_name', 'album_name', 'artists',
            'popularity_category'
        ]

        features_to_drop = [col for col in exclude_features if col in self.df.columns]
        self.df.drop(columns=features_to_drop, inplace=True)

        # Separate features and target
        y = self.df[self.target_variable]
        X = self.df.drop(self.target_variable, axis=1)

        logger.info(f"Features (X): {X.shape}")
        logger.info(f"Target (y): {y.shape}")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        logger.info(f"Splitting data (test_size={test_size})...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test, X, y, output_dir="data/processed"):
        """Save processed data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed data to {output_dir}")

        # Save splits
        X_train.to_parquet(output_dir / 'X_train.parquet', index=False)
        X_test.to_parquet(output_dir / 'X_test.parquet', index=False)
        y_train.to_frame().to_parquet(output_dir / 'y_train.parquet', index=False)
        y_test.to_frame().to_parquet(output_dir / 'y_test.parquet', index=False)

        # Save full dataset
        df_ml_full = pd.concat([X, y], axis=1)
        df_ml_full.to_parquet(output_dir / 'ml_ready_data.parquet', index=False)

        # Save feature info
        feature_info = pd.DataFrame({
            'feature': X.columns,
            'dtype': X.dtypes.astype(str).values,
            'missing': X.isnull().sum().values,
            'unique': X.nunique().values,
            'mean': X.mean().values,
            'std': X.std().values
        })
        feature_info.to_csv(output_dir / 'feature_info.csv', index=False)

        logger.info("Saved all files successfully")
        return output_dir

    def run(self):
        """Run complete feature engineering pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)

        # Load and process
        self.load_data()
        self.create_interaction_features()
        self.encode_features()
        self.scale_features()

        # Prepare and split
        X, y = self.prepare_for_ml()
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Save
        output_dir = self.save_data(X_train, X_test, y_train, y_test, X, y)

        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 80)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.run()

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Target variable: popularity (regression)")
    print("\nFiles saved to data/processed/:")
    print("  ✓ X_train.parquet")
    print("  ✓ X_test.parquet")
    print("  ✓ y_train.parquet")
    print("  ✓ y_test.parquet")
    print("  ✓ ml_ready_data.parquet")
    print("  ✓ feature_info.csv")
