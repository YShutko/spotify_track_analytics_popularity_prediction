"""
Test Script for Improved ML Pipeline with Synthetic Data
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("🧪 TESTING ML PIPELINE WITH SYNTHETIC DATA")
print("="*80)

np.random.seed(42)
n_samples = 1000

print(f"\n📊 Generating {n_samples} synthetic samples...")

data = {
    'danceability': np.random.randn(n_samples),
    'energy': np.random.randn(n_samples),
    'loudness': np.random.randn(n_samples),
    'acousticness': np.random.randn(n_samples),
    'tempo': np.random.randn(n_samples),
    'valence': np.random.randn(n_samples),
    'instrumentalness': np.random.randn(n_samples),
    'duration_min': np.random.uniform(2, 6, n_samples),
    'release_year': np.random.randint(1990, 2024, n_samples)
}

popularity = (
    10 * data['energy'] +
    8 * data['danceability'] +
    5 * data['valence'] +
    np.random.randn(n_samples) * 5 +
    50
)
popularity = np.clip(popularity, 0, 100)
data['popularity'] = popularity

df = pd.DataFrame(data)
output_path = 'cleaned_music_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Synthetic data saved to: {output_path}")
print(f"\nDataset shape: {df.shape}")
print("\n" + "="*80)
print("✅ Now run: python src/improved_ml_pipeline.py")
print("="*80)
