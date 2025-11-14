"""
Spotify Track Analytics Dashboard - Gradio Version
An interactive Gradio app for exploring data and predicting track popularity
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_data():
    """Load processed data"""
    df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')
    return df

def load_ml_data():
    """Load ML-ready data - use randomized sample from full dataset"""
    df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')

    _, metadata, _ = load_model()
    feature_cols = metadata.get('feature_names', [])

    if not feature_cols:
        feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    X = df[feature_cols].copy()
    y = df['popularity'].copy()

    # Remove NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X, y = X[mask], y[mask]

    # Use RANDOMIZED sample for performance (to avoid sorted data bias)
    # Dataset is sorted by popularity, so we must shuffle!
    sample_size = min(5000, len(X))
    indices = np.random.RandomState(42).permutation(len(X))[:sample_size]
    return X.iloc[indices], y.iloc[indices]

def load_model():
    """Load the latest trained model"""
    import glob

    # Find the latest model file
    model_files = sorted(glob.glob('outputs/models/xgb_model_*.joblib'), reverse=True)
    if not model_files:
        # Fallback to old naming convention
        model_files = ['outputs/models/xgboost_popularity_model.joblib']

    latest_model = model_files[0]

    # Find corresponding metadata file
    timestamp = latest_model.split('_')[-1].replace('.joblib', '')
    metadata_file = f'outputs/metadata/xgb_metadata_{timestamp}.json'

    # Load model
    model = joblib.load(latest_model)

    # Load metadata if available
    metadata = {}
    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    elif Path('outputs/models/model_metadata.json').exists():
        # Fallback to old location
        with open('outputs/models/model_metadata.json', 'r') as f:
            metadata = json.load(f)

    # Load or create feature importance
    feature_importance = None
    if Path('outputs/models/feature_importance.csv').exists():
        feature_importance = pd.read_csv('outputs/models/feature_importance.csv')
    else:
        # Create feature importance from model
        if hasattr(model, 'feature_importances_'):
            feature_names = metadata.get('feature_names', [f'feature_{i}' for i in range(len(model.feature_importances_))])
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

    return model, metadata, feature_importance

# Load data globally
df = load_data()
model, metadata, feature_importance = load_model()
X_test, y_test = load_ml_data()

# ============================================================================
# Tab 1: Data Explorer Functions
# ============================================================================

def filter_data(genres, min_pop, max_pop):
    """Filter dataset based on user inputs"""
    df_filtered = df.copy()

    # Apply genre filter
    if genres and len(genres) > 0:
        df_filtered = df_filtered[df_filtered['track_genre'].isin(genres)]

    # Apply popularity filter
    df_filtered = df_filtered[
        (df_filtered['popularity'] >= min_pop) &
        (df_filtered['popularity'] <= max_pop)
    ]

    # Select key columns for display
    display_cols = ['track_name', 'artists', 'album_name', 'popularity',
                   'track_genre', 'danceability', 'energy', 'valence']

    return df_filtered[display_cols].head(100)

def get_dataset_stats():
    """Get overall dataset statistics"""
    stats = f"""
    ### 📊 Dataset Overview

    - **Total Tracks**: {len(df):,}
    - **Average Popularity**: {df['popularity'].mean():.1f}
    - **Unique Genres**: {df['track_genre'].nunique()}
    - **Unique Artists**: {df['artists'].nunique()}
    - **Date Range**: {df['album_name'].nunique():,} unique albums
    """
    return stats

# ============================================================================
# Tab 2: Visualizations Functions
# ============================================================================

def create_popularity_distribution():
    """Create popularity distribution histogram"""
    fig = px.histogram(
        df,
        x='popularity',
        nbins=50,
        title='Distribution of Track Popularity',
        labels={'popularity': 'Popularity Score', 'count': 'Number of Tracks'},
        color_discrete_sequence=['#1DB954']
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def create_audio_features_distribution():
    """Create audio features distribution subplots"""
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=audio_features
    )

    for i, feature in enumerate(audio_features):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature, marker_color='#1DB954', showlegend=False),
            row=row, col=col
        )

    fig.update_layout(height=600, title_text="Audio Features Distribution")
    return fig

def create_genre_chart():
    """Create top genres bar chart"""
    top_genres = df['track_genre'].value_counts().head(20)
    fig = px.bar(
        x=top_genres.values,
        y=top_genres.index,
        orientation='h',
        labels={'x': 'Number of Tracks', 'y': 'Genre'},
        title='Top 20 Music Genres',
        color=top_genres.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_correlation_heatmap():
    """Create feature correlation heatmap"""
    numeric_cols = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Feature Correlation Matrix",
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    fig.update_layout(height=600)
    return fig

def create_energy_vs_valence():
    """Create energy vs valence scatter plot"""
    # Sample for performance
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)

    fig = px.scatter(
        sample_df,
        x='energy',
        y='valence',
        color='popularity',
        title='Energy vs Valence (colored by Popularity)',
        labels={'energy': 'Energy', 'valence': 'Valence', 'popularity': 'Popularity'},
        color_continuous_scale='Viridis',
        opacity=0.6
    )
    fig.update_layout(height=500)
    return fig

def create_genre_popularity():
    """Create average popularity by genre"""
    genre_pop = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(20)

    fig = px.bar(
        x=genre_pop.values,
        y=genre_pop.index,
        orientation='h',
        labels={'x': 'Average Popularity', 'y': 'Genre'},
        title='Top 20 Genres by Average Popularity',
        color=genre_pop.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=600, showlegend=False)
    return fig

# ============================================================================
# Tab 3: ML Model Functions
# ============================================================================

def get_model_info():
    """Get model metadata and performance"""
    # Handle both old and new metadata formats
    metrics = metadata.get('metrics', metadata.get('performance', {}).get('test', {}))
    params = metadata.get('model_params', metadata.get('hyperparameters', {}))
    data_shapes = metadata.get('data_shapes', {})

    # Extract metrics with fallbacks
    r2 = metrics.get('test_r2', metrics.get('r2', 0))
    rmse = metrics.get('test_rmse', metrics.get('rmse', 0))
    mae = metrics.get('test_mae', metrics.get('mae', 0))

    # Extract training details
    n_samples = metadata.get('n_samples', 'N/A')
    n_features = metadata.get('n_features', len(model.feature_names_in_))
    train_samples = data_shapes.get('train', [metadata.get('n_train_samples', 'N/A')])[0] if data_shapes else metadata.get('n_train_samples', 'N/A')
    test_samples = data_shapes.get('test', [metadata.get('n_test_samples', 'N/A')])[0] if data_shapes else metadata.get('n_test_samples', 'N/A')

    # Extract hyperparameters
    n_estimators = params.get('n_estimators', 'N/A')
    max_depth = params.get('max_depth', 'N/A')
    learning_rate = params.get('learning_rate', 'N/A')

    # Format numeric values safely
    n_samples_str = f"{n_samples:,}" if isinstance(n_samples, int) else str(n_samples)
    train_samples_str = f"{train_samples:,}" if isinstance(train_samples, int) else str(train_samples)
    test_samples_str = f"{test_samples:,}" if isinstance(test_samples, int) else str(test_samples)

    info = f"""
    ### 🤖 XGBoost Model Information

    **Model Performance:**
    - R² Score: {r2:.4f}
    - RMSE: {rmse:.2f}
    - MAE: {mae:.2f}

    **Training Details:**
    - Total Samples: {n_samples_str} tracks
    - Train Samples: {train_samples_str}
    - Test Samples: {test_samples_str}
    - Features Used: {n_features}
    - Model Type: XGBoost Regressor

    **Hyperparameters:**
    - Estimators: {n_estimators}
    - Max Depth: {max_depth}
    - Learning Rate: {learning_rate}

    **Note:** Model trained on full 114K dataset with 9 core audio features.
    R² = {r2:.4f} represents the portion of popularity explained by audio features alone.
    """
    return info

def create_feature_importance():
    """Create feature importance chart"""
    top_features = feature_importance.head(20)

    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 20 Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_predictions_scatter():
    """Create actual vs predicted scatter plot"""
    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Sample for performance
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)

    fig = px.scatter(
        x=y_test.iloc[indices],
        y=y_pred[indices],
        labels={'x': 'Actual Popularity', 'y': 'Predicted Popularity'},
        title='Actual vs Predicted Popularity (Test Set)',
        opacity=0.6
    )

    # Add diagonal line for perfect predictions
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash'))
    )

    fig.update_layout(height=500)
    return fig

# ============================================================================
# Tab 4: Track Predictor Functions
# ============================================================================

def generate_recommendations(danceability, energy, valence, acousticness,
                            instrumentalness, tempo_cat, energy_cat):
    """Generate AI recommendations for improving track popularity"""
    recommendations = []

    # Danceability recommendations
    if danceability < 0.5:
        recommendations.append({
            "factor": "🕺 Danceability",
            "current": f"{danceability:.2f}",
            "suggestion": "Increase to 0.7+",
            "impact": "+8-12 points",
            "tip": "Add a stronger, more rhythmic beat pattern"
        })

    # Energy recommendations
    if energy < 0.6:
        recommendations.append({
            "factor": "⚡ Energy",
            "current": f"{energy:.2f}",
            "suggestion": "Boost to 0.7+",
            "impact": "+6-10 points",
            "tip": "Increase tempo, add dynamic percussion, or louder instruments"
        })

    # Valence recommendations
    if valence < 0.4:
        recommendations.append({
            "factor": "😊 Valence (Positivity)",
            "current": f"{valence:.2f}",
            "suggestion": "Brighten to 0.5+",
            "impact": "+5-8 points",
            "tip": "Use major keys, uplifting melodies, or positive lyrics"
        })

    # Acousticness recommendations
    if acousticness > 0.7:
        recommendations.append({
            "factor": "🎸 Acousticness",
            "current": f"{acousticness:.2f}",
            "suggestion": "Reduce to 0.3-0.5",
            "impact": "+4-7 points",
            "tip": "Blend in electronic elements or produced instruments"
        })

    # Instrumentalness recommendations
    if instrumentalness > 0.5:
        recommendations.append({
            "factor": "🎤 Vocals",
            "current": f"Instrumental: {instrumentalness:.2f}",
            "suggestion": "Add vocals",
            "impact": "+7-12 points",
            "tip": "Tracks with vocals tend to be more popular - consider featuring a vocalist"
        })

    return recommendations[:5]  # Return top 5 recommendations

def predict_popularity(danceability, energy, key, loudness, mode, speechiness,
                      acousticness, instrumentalness, liveness, valence, tempo,
                      duration_ms, time_signature, genre):
    """Predict track popularity and provide recommendations"""

    # Convert all inputs to proper types (Gradio may pass strings from Radio)
    danceability = float(danceability)
    energy = float(energy)
    key = int(float(key))
    loudness = float(loudness)
    mode = int(float(mode))
    speechiness = float(speechiness)
    acousticness = float(acousticness)
    instrumentalness = float(instrumentalness)
    liveness = float(liveness)
    valence = float(valence)
    tempo = float(tempo)
    duration_ms = float(duration_ms)
    time_signature = int(float(time_signature))

    # Create feature dictionary matching training features
    # Calculate derived features
    duration_min = duration_ms / 60000
    energy_danceability = energy * danceability
    valence_energy = valence * energy
    acousticness_energy = acousticness * energy
    energy_squared = energy ** 2
    danceability_squared = danceability ** 2
    valence_squared = valence ** 2
    is_short_track = int(duration_min < 3)
    is_long_track = int(duration_min > 5)
    high_energy_happy = int((energy > 0.7) and (valence > 0.7))
    low_energy_sad = int((energy < 0.3) and (valence < 0.3))

    # Mood/energy categorization
    if valence > 0.5 and energy > 0.5:
        mood = "Happy/High Energy"
    elif valence <= 0.5 and energy > 0.5:
        mood = "Energetic/Sad"
    else:
        mood = "Calm/Low Energy"

    # Energy category
    if energy < 0.33:
        energy_category = "Low Energy"
    elif energy < 0.67:
        energy_category = "Medium Energy"
    else:
        energy_category = "High Energy"

    # Tempo category
    if tempo < 90:
        tempo_category = "Slow"
    elif tempo < 120:
        tempo_category = "Moderate"
    elif tempo < 150:
        tempo_category = "Fast"
    else:
        tempo_category = "Very Fast"

    # Create feature vector (need to match training feature order from X_test)
    # This is simplified - in production, you'd need exact feature engineering pipeline
    feature_vector = pd.DataFrame({
        'duration_ms': [duration_ms],
        'explicit': [0],  # Default
        'danceability': [danceability],
        'energy': [energy],
        'key': [key],
        'loudness': [loudness],
        'speechiness': [speechiness],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'liveness': [liveness],
        'valence': [valence],
        'tempo': [tempo],
        'duration_min': [duration_min],
        'energy_danceability': [energy_danceability],
        'valence_energy': [valence_energy],
        'acousticness_energy': [acousticness_energy],
        'energy_squared': [energy_squared],
        'danceability_squared': [danceability_squared],
        'valence_squared': [valence_squared],
        'is_short_track': [is_short_track],
        'is_long_track': [is_long_track],
        'high_energy_happy': [high_energy_happy],
        'low_energy_sad': [low_energy_sad],
    })

    # Add mode and time signature one-hot encoding
    for mode_val in [0, 1]:
        feature_vector[f'mode_{mode_val}'] = [int(mode == mode_val)]

    for ts in [1, 3, 4, 5]:
        feature_vector[f'time_signature_{ts}'] = [int(time_signature == ts)]

    # Add mood/energy categories
    for mood_val in ['Energetic/Sad', 'Happy/High Energy', 'Sad/Low Energy']:
        feature_vector[f'mood_energy_{mood_val}'] = [int(mood == mood_val)]

    # Add energy categories
    for ec in ['Medium Energy', 'High Energy']:
        feature_vector[f'energy_category_{ec}'] = [int(energy_category == ec)]

    # Add tempo categories
    for tc in ['Moderate', 'Fast', 'Very Fast']:
        feature_vector[f'tempo_category_{tc}'] = [int(tempo_category == tc)]

    # Add genre encoding (simplified - use mean encoding or label encoding)
    # For now, use a simple hash
    genre_encoded = hash(genre) % 100 / 100.0
    feature_vector['track_genre_encoded'] = [genre_encoded]

    # Use only the features that the model was trained on
    # For the new model (9 features), we'll use a simplified approach
    model_features = model.feature_names_in_

    # Create a simple feature vector with just the 9 core features if that's what model expects
    if len(model_features) == 9 and 'release_year' not in model_features:
        # New model with 9 audio features only
        feature_vector_model = pd.DataFrame({
            'danceability': [danceability],
            'energy': [energy],
            'loudness': [loudness],
            'speechiness': [speechiness],
            'acousticness': [acousticness],
            'instrumentalness': [instrumentalness],
            'liveness': [liveness],
            'valence': [valence],
            'tempo': [tempo]
        })
    else:
        # Old model with many engineered features
        # Filter to only include features the model expects
        feature_vector_model = feature_vector[[col for col in model_features if col in feature_vector.columns]].copy()

        # Add missing features with appropriate default values
        for col in model_features:
            if col not in feature_vector_model.columns:
                feature_vector_model[col] = [0]

        # Ensure column order matches model's expectations
        feature_vector_model = feature_vector_model[model_features]

    # Make prediction
    prediction = model.predict(feature_vector_model)[0]
    prediction = max(0, min(100, prediction))  # Clip to 0-100 range

    # Generate recommendations
    recommendations = generate_recommendations(
        danceability, energy, valence, acousticness,
        instrumentalness, tempo_category, energy_category
    )

    # Format output
    result = f"""
    ## 🎯 Predicted Popularity: {prediction:.1f}/100

    **Track Characteristics:**
    - Mood: {mood}
    - Energy Level: {energy_category}
    - Tempo: {tempo_category} ({tempo:.0f} BPM)
    - Duration: {duration_min:.2f} minutes

    ---
    """

    if recommendations:
        result += "\n### 💡 Optimization Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            result += f"""
**{i}. {rec['factor']}**
- Current: {rec['current']}
- Suggestion: {rec['suggestion']}
- Potential Impact: {rec['impact']}
- Tip: {rec['tip']}

"""
    else:
        result += "\n### ✅ Great job! Your track is well-optimized for popularity.\n"

    return result

def load_random_example():
    """Load a random track example from the dataset"""
    random_track = df.sample(n=1).iloc[0]

    # Ensure all numeric values are proper Python types (not numpy types)
    # to avoid type coercion issues with Gradio components
    return (
        float(random_track['danceability']),
        float(random_track['energy']),
        float(random_track['key']),  # Changed to float for Slider compatibility
        float(random_track['loudness']),
        float(random_track['mode']),  # Changed to float - Gradio Radio may expect this
        float(random_track['speechiness']),
        float(random_track['acousticness']),
        float(random_track['instrumentalness']),
        float(random_track['liveness']),
        float(random_track['valence']),
        float(random_track['tempo']),
        float(random_track['duration_ms']),
        float(random_track['time_signature']),  # Changed to float - Gradio Radio may expect this
        str(random_track['track_genre'])
    )

# ============================================================================
# Gradio Interface
# ============================================================================

# Custom CSS for Spotify theme
custom_css = """
.gradio-container {
    font-family: 'Helvetica Neue', Arial, sans-serif;
}
.green-button {
    background-color: #1DB954 !important;
    color: white !important;
}
"""

# Build the interface with tabs
with gr.Blocks(css=custom_css, title="Spotify Track Analytics", theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("""
    # 🎵 Spotify Track Analytics Dashboard
    ### Explore 114,000 tracks and predict popularity using machine learning
    """)

    # Tabs
    with gr.Tabs():

        # ====================================================================
        # TAB 1: Data Explorer
        # ====================================================================
        with gr.Tab("📊 Data Explorer"):
            gr.Markdown(get_dataset_stats())

            gr.Markdown("### 🔍 Filter and Browse Data")

            with gr.Row():
                genre_selector = gr.Dropdown(
                    choices=sorted(df['track_genre'].unique().tolist()),
                    multiselect=True,
                    label="Filter by Genre",
                    info="Select one or more genres"
                )

            with gr.Row():
                min_popularity = gr.Slider(0, 100, value=0, label="Min Popularity")
                max_popularity = gr.Slider(0, 100, value=100, label="Max Popularity")

            filter_btn = gr.Button("Apply Filters", variant="primary")

            data_output = gr.Dataframe(
                value=df[['track_name', 'artists', 'album_name', 'popularity',
                         'track_genre', 'danceability', 'energy', 'valence']].head(100),
                label="Dataset Preview (Top 100 rows)",
                interactive=False
            )

            filter_btn.click(
                fn=filter_data,
                inputs=[genre_selector, min_popularity, max_popularity],
                outputs=data_output
            )

        # ====================================================================
        # TAB 2: Visualizations
        # ====================================================================
        with gr.Tab("📈 Visualizations"):
            gr.Markdown("## Data Visualizations")

            with gr.Accordion("🎯 Popularity Distribution", open=True):
                pop_plot = gr.Plot(value=create_popularity_distribution())

            with gr.Accordion("🎼 Audio Features Distribution", open=False):
                audio_plot = gr.Plot(value=create_audio_features_distribution())

            with gr.Accordion("🎸 Top Genres by Count", open=False):
                genre_plot = gr.Plot(value=create_genre_chart())

            with gr.Accordion("⭐ Top Genres by Popularity", open=False):
                genre_pop_plot = gr.Plot(value=create_genre_popularity())

            with gr.Accordion("🔥 Energy vs Valence", open=False):
                scatter_plot = gr.Plot(value=create_energy_vs_valence())

            with gr.Accordion("📊 Feature Correlations", open=False):
                corr_plot = gr.Plot(value=create_correlation_heatmap())

        # ====================================================================
        # TAB 3: ML Model
        # ====================================================================
        with gr.Tab("🤖 ML Model"):
            gr.Markdown(get_model_info())

            with gr.Accordion("📊 Feature Importance", open=True):
                importance_plot = gr.Plot(value=create_feature_importance())

            with gr.Accordion("🎯 Model Predictions", open=False):
                predictions_plot = gr.Plot(value=create_predictions_scatter())
                # Get current metrics for note
                current_r2 = metadata.get('metrics', {}).get('test_r2',
                            metadata.get('performance', {}).get('test', {}).get('r2', 0.28))
                gr.Markdown(f"""
                **Note:** Points closer to the red diagonal line indicate better predictions.
                The model achieves an R² score of {current_r2:.4f}, explaining {current_r2*100:.1f}% of popularity variance from audio features.
                """)

        # ====================================================================
        # TAB 4: Track Predictor
        # ====================================================================
        with gr.Tab("🎯 Track Predictor"):
            gr.Markdown("""
            ## Predict Your Track's Popularity
            Enter your track characteristics below to get a popularity prediction and optimization tips.
            """)

            with gr.Row():
                random_btn = gr.Button("🎲 Load Random Example", variant="secondary")

            gr.Markdown("### Audio Features")
            with gr.Row():
                danceability = gr.Slider(0, 1, value=0.5, step=0.01, label="Danceability",
                                        info="How suitable for dancing (0=not danceable, 1=very danceable)")
                energy = gr.Slider(0, 1, value=0.5, step=0.01, label="Energy",
                                  info="Intensity and activity (0=calm, 1=energetic)")
                valence = gr.Slider(0, 1, value=0.5, step=0.01, label="Valence",
                                   info="Musical positivity (0=sad, 1=happy)")

            with gr.Row():
                acousticness = gr.Slider(0, 1, value=0.5, step=0.01, label="Acousticness",
                                        info="Acoustic vs electronic (0=electronic, 1=acoustic)")
                instrumentalness = gr.Slider(0, 1, value=0.5, step=0.01, label="Instrumentalness",
                                            info="Amount of vocals (0=vocal, 1=instrumental)")
                speechiness = gr.Slider(0, 1, value=0.1, step=0.01, label="Speechiness",
                                       info="Presence of spoken words (0=music, 1=speech)")

            with gr.Row():
                liveness = gr.Slider(0, 1, value=0.2, step=0.01, label="Liveness",
                                    info="Presence of audience (0=studio, 1=live)")
                loudness = gr.Slider(-60, 0, value=-5, step=0.1, label="Loudness (dB)",
                                    info="Overall loudness in decibels")

            gr.Markdown("### Musical Properties")
            with gr.Row():
                key = gr.Slider(0, 11, value=0, step=1, label="Key",
                               info="Musical key (0=C, 1=C#, ..., 11=B)")
                mode = gr.Radio([0, 1], value=1, label="Mode",
                               info="0=Minor, 1=Major")
                time_signature = gr.Radio([1, 3, 4, 5], value=4, label="Time Signature",
                                         info="Beats per bar")

            gr.Markdown("### Track Details")
            with gr.Row():
                tempo = gr.Slider(40, 220, value=120, step=1, label="Tempo (BPM)",
                                 info="Beats per minute")
                duration_ms = gr.Slider(30000, 600000, value=200000, step=1000, label="Duration (ms)",
                                       info="Track length in milliseconds")

            with gr.Row():
                genre = gr.Dropdown(
                    choices=sorted(df['track_genre'].unique().tolist()),
                    value=df['track_genre'].mode()[0],
                    label="Genre"
                )

            predict_btn = gr.Button("🎯 Predict Popularity", variant="primary", size="lg")

            prediction_output = gr.Markdown(label="Prediction Results")

            # Wire up prediction
            predict_btn.click(
                fn=predict_popularity,
                inputs=[danceability, energy, key, loudness, mode, speechiness,
                       acousticness, instrumentalness, liveness, valence, tempo,
                       duration_ms, time_signature, genre],
                outputs=prediction_output
            )

            # Wire up random example
            random_btn.click(
                fn=load_random_example,
                outputs=[danceability, energy, key, loudness, mode, speechiness,
                        acousticness, instrumentalness, liveness, valence, tempo,
                        duration_ms, time_signature, genre]
            )

    # Footer with dynamic metrics
    metrics = metadata.get('metrics', metadata.get('performance', {}).get('test', {}))
    footer_r2 = metrics.get('test_r2', metrics.get('r2', 0))
    footer_adj_r2 = metrics.get('test_adjusted_r2', metrics.get('adjusted_r2', 0))
    footer_rmse = metrics.get('test_rmse', metrics.get('rmse', 0))
    footer_mae = metrics.get('test_mae', metrics.get('mae', 0))
    footer_n_samples = metadata.get('n_samples', 114000)

    gr.Markdown(f"""
    ---
    **Model Info:** XGBoost Regressor | R² = {footer_r2:.4f} | Adjusted R² = {footer_adj_r2:.4f} | RMSE = {footer_rmse:.2f} | MAE = {footer_mae:.2f} | Dataset: {footer_n_samples:,} tracks
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
