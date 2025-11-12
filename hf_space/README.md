---
title: Spotify Track Analytics
emoji: 🎵
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# 🎵 Spotify Track Analytics

Explore 114,000 Spotify tracks and predict popularity using machine learning!

## Features

- **📊 Data Explorer**: Browse and filter tracks by genre and popularity
- **📈 Visualizations**: Interactive charts showing audio features, genres, and correlations
- **🤖 ML Model**: XGBoost model with R² = 0.39, trained on 37 features
- **🎯 Track Predictor**: Predict popularity for your tracks + get AI recommendations

## How to Use

1. **Data Explorer Tab**: Filter by genre and popularity range to browse tracks
2. **Visualizations Tab**: Explore interactive charts (click accordions to expand)
3. **ML Model Tab**: View model performance and feature importance
4. **Track Predictor Tab**:
   - Click "🎲 Load Random Example" to see a sample track
   - Or manually adjust sliders to input your track's characteristics
   - Click "🎯 Predict Popularity" to get results and optimization tips

## Model Details

- **Algorithm**: XGBoost Regressor
- **Performance**: R² = 0.3882, RMSE = 17.38, MAE = 13.00
- **Features**: 37 engineered features including audio characteristics, mood categories, and interaction terms
- **Dataset**: 114,000 tracks across 114 genres

## Track Predictor Tips

The AI recommendation engine analyzes your track and suggests improvements:
- Low danceability? Add rhythmic beats (+8-12 popularity points)
- Low energy? Increase tempo and dynamics (+6-10 points)
- Low valence? Use major keys and uplifting melodies (+5-8 points)
- High instrumentalness? Add vocals (+7-12 points)
- High acousticness? Blend electronic elements (+4-7 points)

## Dataset

Based on the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle.

## GitHub Repository

Full source code, documentation, and ML pipeline: [spotify_track_analytics_popularity_prediction](https://github.com/YShutko/spotify_track_analytics_popularity_prediction)

---

Built with Gradio 🚀 | Model: XGBoost | Dataset: 114K tracks
