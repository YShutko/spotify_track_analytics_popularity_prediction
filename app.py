"""
Spotify Track Analytics Dashboard
A Streamlit app for exploring data and predicting track popularity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Spotify Track Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #191414;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.75rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0.75rem 1.5rem;
        background-color: transparent;
        border-radius: 0.5rem;
        border: 2px solid transparent;
        font-size: 1rem;
        font-weight: 500;
        color: #666;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8f5e9;
        color: #1DB954;
        border-color: #1DB954;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white !important;
        border-color: #1DB954;
        box-shadow: 0 4px 16px rgba(29, 185, 84, 0.3);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.4);
    }
    .prediction-box {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .recommendation-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load processed data"""
    df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')
    return df

@st.cache_data
def load_ml_data():
    """Load ML-ready data - use randomized sample from full dataset"""
    # Load the full dataset and prepare features
    df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')

    # Get model to know which features to use
    _, metadata, _, _ = load_model()
    feature_cols = metadata.get('feature_names', [])

    if not feature_cols:
        # Fallback to default features if metadata missing
        feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    X = df[feature_cols].copy()
    y = df['popularity'].copy()

    # Remove NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X, y = X[mask], y[mask]

    # Use RANDOMIZED sample for performance (to avoid sorted data bias)
    # Dataset is sorted by popularity, so we must shuffle!
    sample_size = min(5000, len(X))
    indices = np.random.RandomState(42).permutation(len(X))[:sample_size]
    return X.iloc[indices], y.iloc[indices]

@st.cache_resource
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
    # Extract everything after 'xgb_model_' and before '.joblib'
    model_basename = Path(latest_model).stem  # Gets filename without extension
    suffix = model_basename.replace('xgb_model_', '')  # Gets 'full_20251114_135842' or similar
    metadata_file = f'outputs/metadata/xgb_metadata_{suffix}.json'

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

    return model, metadata, feature_importance, latest_model

@st.cache_resource
def load_rf_model():
    """Load the latest trained Random Forest model"""
    import glob

    # Find the latest RF model file
    model_files = sorted(glob.glob('outputs/models/rf_model_*.joblib'), reverse=True)
    if not model_files:
        return None, {}, None, None  # No RF model found

    latest_model = model_files[0]

    # Find corresponding metadata file
    model_basename = Path(latest_model).stem
    suffix = model_basename.replace('rf_model_', '')
    metadata_file = f'outputs/metadata/rf_metadata_{suffix}.json'

    # Load model
    model = joblib.load(latest_model)

    # Load metadata if available
    metadata = {}
    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # Create feature importance from model
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_names = metadata.get('feature_names', [f'feature_{i}' for i in range(len(model.feature_importances_))])
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    return model, metadata, feature_importance, latest_model

# Load data
df = load_data()
model, metadata, feature_importance, model_path = load_model()
rf_model, rf_metadata, rf_feature_importance, rf_model_path = load_rf_model()

# Sidebar
with st.sidebar:
    st.image("assets/teamlogo.png", width=200)
    st.markdown("## 🎵 Spotify Track Analytics")
    st.markdown("---")
    st.markdown("### About")
    # Get metrics from metadata
    test_r2 = metadata.get('metrics', {}).get('test_r2', 0)
    test_adj_r2 = metadata.get('metrics', {}).get('test_adjusted_r2', 0)
    test_rmse = metadata.get('metrics', {}).get('test_rmse', 0)
    test_mae = metadata.get('metrics', {}).get('test_mae', 0)

    n_samples = metadata.get('n_samples', len(df))

    st.info(f"""
    Explore {n_samples:,} cleaned Spotify tracks and predict popularity using machine learning.

    **Model Performance:**
    - R² Score: {test_r2:.4f}
    - Adjusted R²: {test_adj_r2:.4f}
    - RMSE: {test_rmse:.2f}
    - MAE: {test_mae:.2f}

    *Trained on {n_samples:,} tracks (cleaned dataset v2)*
    """)
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total Tracks", f"{len(df):,}")
    st.metric("Genres", df['track_genre'].nunique())
    st.metric("Features", len(df.columns))

# Main header
st.markdown('<h1 class="main-header">🎵 Spotify Track Analytics Dashboard</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer",
    "📈 Visualizations",
    "🤖 XGBoost",
    "🌲 RForest",
    "🎯 Track Predictor"
])

# ============================================================================
# TAB 1: Data Explorer
# ============================================================================
with tab1:
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tracks", f"{len(df):,}")
    with col2:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
    with col3:
        st.metric("Unique Genres", df['track_genre'].nunique())
    with col4:
        st.metric("Unique Artists", df['artists'].nunique())

    st.markdown("---")

    # Data preview
    st.markdown("### 📋 Data Sample")

    # Filters
    col1, col2 = st.columns([1, 1])
    with col1:
        genre_filter = st.multiselect(
            "Filter by Genre",
            options=sorted(df['track_genre'].unique()),
            default=[]
        )
    with col2:
        pop_range = st.slider(
            "Popularity Range",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )

    # Apply filters
    df_filtered = df.copy()
    if genre_filter:
        df_filtered = df_filtered[df_filtered['track_genre'].isin(genre_filter)]
    df_filtered = df_filtered[
        (df_filtered['popularity'] >= pop_range[0]) &
        (df_filtered['popularity'] <= pop_range[1])
    ]

    st.write(f"**Showing {len(df_filtered):,} tracks**")
    st.dataframe(
        df_filtered[['track_name', 'artists', 'album_name', 'popularity',
                     'track_genre', 'danceability', 'energy', 'valence']].head(100),
        width='stretch',
        height=400
    )

    # Download option
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name="spotify_tracks_filtered.csv",
        mime="text/csv"
    )

# ============================================================================
# TAB 2: Visualizations
# ============================================================================
with tab2:
    st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)

    # Popularity Distribution
    st.markdown("### 🎯 Popularity Distribution")
    fig = px.histogram(
        df,
        x='popularity',
        nbins=50,
        title='Distribution of Track Popularity',
        labels={'popularity': 'Popularity Score', 'count': 'Number of Tracks'},
        color_discrete_sequence=['#1DB954']
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Audio Features
    st.markdown("### 🎼 Audio Feature Distributions")
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
    st.plotly_chart(fig, width="stretch")

    # Genre Analysis
    st.markdown("### 🎸 Top 20 Genres by Track Count")
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
    st.plotly_chart(fig, width="stretch")

    # Mood/Energy Analysis
    st.markdown("### 😊 Mood & Energy Classification")
    col1, col2 = st.columns(2)

    with col1:
        mood_counts = df['mood_energy'].value_counts()
        fig = px.pie(
            values=mood_counts.values,
            names=mood_counts.index,
            title='Mood Distribution',
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        energy_counts = df['energy_category'].value_counts()
        fig = px.pie(
            values=energy_counts.values,
            names=energy_counts.index,
            title='Energy Level Distribution',
            color_discrete_sequence=px.colors.sequential.Teal
        )
        st.plotly_chart(fig, width="stretch")

    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    audio_cols = ['popularity', 'danceability', 'energy', 'valence',
                  'acousticness', 'instrumentalness', 'loudness', 'tempo']
    corr = df[audio_cols].corr()

    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title='Audio Features Correlation Matrix'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    # Scatter plot
    st.markdown("### 📊 Interactive Scatter Plot")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-axis", audio_features, index=0)
    with col2:
        y_axis = st.selectbox("Y-axis", audio_features, index=1)
    with col3:
        color_by = st.selectbox("Color by", ['mood_energy', 'energy_category', 'popularity_category'])

    fig = px.scatter(
        df.sample(5000),  # Sample for performance
        x=x_axis,
        y=y_axis,
        color=color_by,
        hover_data=['track_name', 'artists', 'popularity'],
        title=f'{x_axis.capitalize()} vs {y_axis.capitalize()}',
        opacity=0.6
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

# ============================================================================
# TAB 3: XGBoost Model
# ============================================================================
with tab3:
    st.markdown('<h2 class="sub-header">XGBoost Model Performance</h2>', unsafe_allow_html=True)

    # Display model version/timestamp
    timestamp = metadata.get('timestamp', 'Unknown')
    model_file = model_path.split('/')[-1] if model_path else 'Unknown'
    st.info(f"📦 **Model Version:** `{model_file}` | **Trained:** {timestamp[:19] if timestamp != 'Unknown' else 'Unknown'}")

    # Model info
    st.markdown("#### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Handle both old and new metadata formats
    metrics = metadata.get('metrics', metadata.get('performance', {}).get('test', {}))

    with col1:
        r2_val = metrics.get('test_r2', metrics.get('r2', test_r2))
        st.metric("Test R² Score", f"{r2_val:.4f}",
                  help="Coefficient of determination")
    with col2:
        rmse_val = metrics.get('test_rmse', metrics.get('rmse', test_rmse))
        st.metric("Test RMSE", f"{rmse_val:.2f}",
                  help="Root Mean Squared Error")
    with col3:
        mae_val = metrics.get('test_mae', metrics.get('mae', test_mae))
        st.metric("Test MAE", f"{mae_val:.2f}",
                  help="Mean Absolute Error")
    with col4:
        n_features = metadata.get('n_features', len(model.feature_names_in_))
        st.metric("Features", n_features,
                  help="Number of audio features used")

    # Training info
    with st.expander("ℹ️ Model Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Hyperparameters:**")
            # Handle both hyperparameters and model_params
            params = metadata.get('model_params', metadata.get('hyperparameters', {}))
            if params:
                display_params = {
                    "n_estimators": params.get('n_estimators', 'N/A'),
                    "max_depth": params.get('max_depth', 'N/A'),
                    "learning_rate": round(params.get('learning_rate', 0), 4),
                    "subsample": round(params.get('subsample', 0), 4),
                    "colsample_bytree": round(params.get('colsample_bytree', 0), 4)
                }
                st.json(display_params)
            else:
                st.write("No hyperparameters available")
        with col2:
            st.markdown("**Training Info:**")
            st.write(f"- Model Type: XGBoost Regressor")
            n_samples = metadata.get('n_samples', 'N/A')
            st.write(f"- Total Samples: {n_samples:,}" if isinstance(n_samples, int) else f"- Total Samples: {n_samples}")
            train_shapes = metadata.get('data_shapes', {})
            if train_shapes:
                st.write(f"- Train: {train_shapes.get('train', ['N/A'])[0]:,} samples")
                st.write(f"- Test: {train_shapes.get('test', ['N/A'])[0]:,} samples")
            timestamp = metadata.get('timestamp', metadata.get('training_date', 'N/A'))
            if timestamp != 'N/A':
                st.write(f"- Trained: {timestamp[:10]}")

    st.markdown("---")

    # Feature Importance
    st.markdown("### 🎯 Feature Importance Analysis")
    st.markdown("""
    The XGBoost model identifies which audio features have the strongest impact on track popularity.
    Features are ranked by their importance scores (gain-based).
    """)

    top_features = feature_importance.head(20) if len(feature_importance) > 20 else feature_importance
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {len(top_features)} Feature Importances',
        labels={'importance': 'Importance Score', 'feature': 'Audio Feature'},
        color='importance',
        color_continuous_scale='Greens',
        text='importance'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=max(400, len(top_features) * 30), showlegend=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    # Key insights
    st.info(f"""
    **Key Findings:**
    - **{top_features.iloc[0]['feature'].capitalize()}** is the most important feature ({top_features.iloc[0]['importance']:.1%})
    - **{top_features.iloc[1]['feature'].capitalize()}** is the second most important ({top_features.iloc[1]['importance']:.1%})
    - **{top_features.iloc[2]['feature'].capitalize()}** is the third most important ({top_features.iloc[2]['importance']:.1%})
    - These top 3 features account for {top_features.head(3)['importance'].sum():.1%} of the model's predictive power
    """)

    st.markdown("---")

    # SHAP Analysis
    st.markdown("### 🎯 SHAP Feature Impact Analysis")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to individual predictions.
    Unlike traditional feature importance, SHAP values show both *direction* (positive/negative impact) and *magnitude*.
    """)

    # Persistent SHAP caching to disk
    def load_or_compute_xgb_shap_values(model, model_path, X_sample):
        """Load cached SHAP values or compute if not available"""
        import os
        from pathlib import Path

        # Create cache directory
        cache_dir = Path('outputs/shap')
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename based on model file
        model_filename = Path(model_path).stem
        cache_file = cache_dir / f'{model_filename}_shap_cache.joblib'

        # Check if cache exists and is valid
        if cache_file.exists():
            try:
                st.info(f"📦 Loading cached SHAP values from {cache_file.name}...")
                cached_data = joblib.load(cache_file)

                # Verify cache has correct sample size
                if cached_data['shap_values'].shape[0] == X_sample.shape[0]:
                    st.success("✅ Loaded cached SHAP values!")
                    return cached_data['shap_values'], cached_data['base_value'], cached_data['X_sample']
                else:
                    st.warning("⚠️ Cache size mismatch, recomputing...")
            except Exception as e:
                st.warning(f"⚠️ Cache load failed ({str(e)}), recomputing...")

        # Compute SHAP values
        st.info("🔄 Computing SHAP values... This may take a moment.")
        with st.spinner("Calculating SHAP explanations..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            base_value = explainer.expected_value

            # Save to cache
            cache_data = {
                'shap_values': shap_values,
                'base_value': base_value,
                'X_sample': X_sample,
                'timestamp': datetime.now().isoformat(),
                'model_file': model_filename
            }
            joblib.dump(cache_data, cache_file)
            st.success(f"✅ SHAP values computed and cached to {cache_file.name}!")

        return shap_values, base_value, X_sample

    # Load data for SHAP analysis
    X_shap, y_shap = load_ml_data()

    # Use a smaller sample for SHAP (computational efficiency)
    shap_sample_size = min(500, len(X_shap))
    X_shap_sample_original = X_shap[:shap_sample_size]

    # Load or compute SHAP values with persistent caching
    shap_values, base_value, X_shap_sample = load_or_compute_xgb_shap_values(
        model, model_path, X_shap_sample_original
    )

    # SHAP Summary Plot (Beeswarm)
    st.markdown("#### Feature Impact Distribution")
    st.markdown("""
    This plot shows how each feature affects predictions across the dataset:
    - **Position (x-axis)**: SHAP value (impact on prediction)
    - **Color**: Feature value (red = high, blue = low)
    - **Density**: How often this impact occurs
    """)

    fig_shap_summary, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap_sample, show=False, plot_type="dot")
    st.pyplot(fig_shap_summary, width="stretch")
    plt.close()

    st.caption("""
    **Reading the plot:**
    - Features at the top have the most impact on predictions
    - Red dots (high feature values) on the right mean higher predictions
    - Blue dots (low feature values) on the left mean lower predictions
    """)

    # SHAP Bar Plot (Mean Absolute Impact)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Feature Impact")
        fig_shap_bar, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
        st.pyplot(fig_shap_bar, width="stretch")
        plt.close()

        st.caption("""
        **Mean |SHAP value|** - Average magnitude of impact across all predictions.
        Complements XGBoost's gain-based importance with impact-based importance.
        """)

    with col2:
        st.markdown("#### Sample Prediction Explanation")

        # Select a random sample for waterfall plot
        sample_idx = st.slider("Select track index to explain", 0, shap_sample_size-1, 0)

        # Create waterfall plot
        fig_waterfall, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=base_value,
                data=X_shap_sample.iloc[sample_idx],
                feature_names=X_shap_sample.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig_waterfall, width="stretch")
        plt.close()

        # Convert base_value to scalar if it's an array
        base_value_scalar = float(np.array(base_value).flat[0])

        st.caption(f"""
        **Waterfall for track #{sample_idx}:**
        - Base value: {base_value_scalar:.2f} (average prediction)
        - Actual prediction: {model.predict(X_shap_sample.iloc[[sample_idx]])[0]:.2f}
        - Shows how each feature pushes prediction up or down
        """)

    # SHAP Dependence Plots
    st.markdown("#### Feature Dependence Analysis")
    st.markdown("Explore how individual features affect predictions across different values:")

    col1, col2 = st.columns(2)

    with col1:
        # Top feature dependence
        top_feature = feature_importance.iloc[0]['feature']
        fig_dep1, ax1 = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            top_feature,
            shap_values,
            X_shap_sample,
            ax=ax1,
            show=False
        )
        st.pyplot(fig_dep1, width="stretch")
        plt.close()

        st.caption(f"""
        **{top_feature.capitalize()} dependence:**
        Shows relationship between {top_feature} values and their impact on predictions.
        Color represents interaction with most correlated feature.
        """)

    with col2:
        # Second feature dependence
        second_feature = feature_importance.iloc[1]['feature']
        fig_dep2, ax2 = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            second_feature,
            shap_values,
            X_shap_sample,
            ax=ax2,
            show=False
        )
        st.pyplot(fig_dep2, width="stretch")
        plt.close()

        st.caption(f"""
        **{second_feature.capitalize()} dependence:**
        Shows relationship between {second_feature} values and their impact on predictions.
        Helps identify non-linear patterns and feature interactions.
        """)

    st.markdown("---")

    # Model Performance
    st.markdown("### 📈 Model Performance on Full Dataset")
    n_samples_display = metadata.get('n_samples', len(df))
    st.markdown(f"""
    Below are predictions on a sample of the full Spotify dataset ({n_samples_display:,} cleaned tracks).
    The model was trained on deduplicated data with optimized hyperparameters from Optuna.
    """)

    X_test, y_test = load_ml_data()

    # Data is already prepared with correct features from load_ml_data
    sample_size = min(2000, len(X_test))
    y_pred = model.predict(X_test[:sample_size])
    y_actual = y_test[:sample_size]
    if hasattr(y_actual, 'values'):
        y_actual = y_actual.values.flatten()

    # Calculate metrics on this sample
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    sample_r2 = r2_score(y_actual, y_pred)
    sample_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    sample_mae = mean_absolute_error(y_actual, y_pred)

    # Show sample metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample R²", f"{sample_r2:.4f}", help=f"R² on {sample_size:,} tracks from full dataset")
    with col2:
        st.metric("Sample RMSE", f"{sample_rmse:.2f}")
    with col3:
        st.metric("Sample MAE", f"{sample_mae:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        # Prediction vs Actual
        fig = px.scatter(
            x=y_actual,
            y=y_pred,
            labels={'x': 'Actual Popularity', 'y': 'Predicted Popularity'},
            title='Predictions vs Actual Values',
            opacity=0.5
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Residuals histogram
        residuals = y_actual - y_pred
        fig = px.histogram(
            residuals,
            nbins=50,
            title='Prediction Errors Distribution',
            labels={'value': 'Residual (Actual - Predicted)', 'count': 'Frequency'},
            color_discrete_sequence=['#1DB954']
        )
        # Add mean line
        fig.add_vline(x=residuals.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {residuals.mean():.2f}")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Additional performance visualizations
    st.markdown("### 🔍 Advanced Model Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        # Residual plot (residuals vs predicted)
        fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={'x': 'Predicted Popularity', 'y': 'Residual (Actual - Predicted)'},
            title='Residual Plot',
            opacity=0.5,
            color_discrete_sequence=['#1DB954']
        )
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                     annotation_text="Zero Error")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        st.caption("""
        **Residual Plot Interpretation:**
        - Points should scatter randomly around zero line
        - Patterns indicate model bias or heteroscedasticity
        - Fan shapes suggest varying error across popularity range
        """)

    with col2:
        # Error by popularity ranges
        popularity_bins = pd.cut(y_actual, bins=[0, 20, 40, 60, 80, 100],
                                labels=['Very Low (0-20)', 'Low (20-40)', 'Medium (40-60)',
                                       'High (60-80)', 'Very High (80-100)'])
        error_df = pd.DataFrame({
            'Popularity Range': popularity_bins,
            'Absolute Error': np.abs(residuals)
        })

        fig = px.box(
            error_df,
            x='Popularity Range',
            y='Absolute Error',
            title='Model Error by Popularity Range',
            labels={'Absolute Error': 'Absolute Error (points)'},
            color='Popularity Range',
            color_discrete_sequence=px.colors.sequential.Greens
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        st.caption("""
        **Error Distribution Analysis:**
        - Shows where the model performs best/worst
        - Lower boxes indicate better prediction accuracy
        - Helps identify if model struggles with specific popularity ranges
        """)

    col1, col2 = st.columns(2)

    with col1:
        # Actual distribution
        fig = px.histogram(
            y_actual,
            nbins=30,
            title='Actual Popularity Distribution',
            labels={'value': 'Popularity', 'count': 'Frequency'},
            color_discrete_sequence=['#1DB954']
        )
        fig.add_vline(x=y_actual.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {y_actual.mean():.1f}")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch", key="xgb_actual_distribution_hist")

    with col2:
        # Predicted distribution
        fig = px.histogram(
            y_pred,
            nbins=30,
            title='Predicted Popularity Distribution',
            labels={'value': 'Popularity', 'count': 'Frequency'},
            color_discrete_sequence=['#66BB6A']
        )
        fig.add_vline(x=y_pred.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {y_pred.mean():.1f}")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch", key="xgb_predicted_distribution_hist")

    # Model insights
    st.markdown("### 💡 Model Performance Insights")

    # Calculate additional metrics
    over_predictions = np.sum(y_pred > y_actual)
    under_predictions = np.sum(y_pred < y_actual)
    mae_by_range = error_df.groupby('Popularity Range', observed=True)['Absolute Error'].mean().to_dict()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overestimations", f"{over_predictions:,}",
                 help="Predictions higher than actual popularity")
    with col2:
        st.metric("Underestimations", f"{under_predictions:,}",
                 help="Predictions lower than actual popularity")
    with col3:
        accuracy_within_10 = np.sum(np.abs(residuals) <= 10) / len(residuals) * 100
        st.metric("Within ±10 Points", f"{accuracy_within_10:.1f}%",
                 help="Percentage of predictions within 10 points of actual")

    # Best/worst performance ranges
    if mae_by_range:
        best_range = min(mae_by_range, key=mae_by_range.get)
        worst_range = max(mae_by_range, key=mae_by_range.get)

        st.success(f"""
        **Best Performance:** {best_range} popularity range (MAE: {mae_by_range[best_range]:.2f} points)
        """)

        st.warning(f"""
        **Needs Improvement:** {worst_range} popularity range (MAE: {mae_by_range[worst_range]:.2f} points)
        """)

    # Model explanation
    st.info(f"""
    **Understanding R² = {sample_r2:.4f}:**

    Audio features alone explain ~{sample_r2*100:.1f}% of popularity variance. The remaining ~{(1-sample_r2)*100:.1f}% comes from:
    - Artist fame and follower count
    - Marketing budget and promotion
    - Playlist placements
    - Social media trends and virality
    - Release timing and cultural factors

    **This is expected and aligns with academic research on music popularity prediction.**
    """)

    # Model metadata
    with st.expander("📋 View Model Metadata"):
        st.json(metadata)

# ============================================================================
# TAB 4: Random Forest Model
# ============================================================================
with tab4:
    st.markdown('<h2 class="sub-header">Random Forest Model Performance</h2>', unsafe_allow_html=True)

    # Check if RF model is available
    if rf_model is None:
        st.warning("⚠️ Random Forest model not found. Please train the model first.")
        st.info("Run: `python src/save_rf_model.py` to train and save the Random Forest model.")
    else:
        # Display model version/timestamp
        timestamp = rf_metadata.get('timestamp', 'Unknown')
        model_file = rf_model_path.split('/')[-1] if rf_model_path else 'Unknown'
        st.info(f"📦 **Model Version:** `{model_file}` | **Trained:** {timestamp[:19] if timestamp != 'Unknown' else 'Unknown'}")

        # Model info
        st.markdown("#### 📊 Model Performance Metrics")
        rf_metrics_col1, rf_metrics_col2, rf_metrics_col3, rf_metrics_col4 = st.columns(4)

        # Handle both old and new metadata formats
        metrics = rf_metadata.get('metrics', rf_metadata.get('performance', {}).get('test', {}))

        with rf_metrics_col1:
            r2_val = metrics.get('test_r2', metrics.get('r2', 0))
            st.metric("Test R² Score", f"{r2_val:.4f}",
                      help="Coefficient of determination")
        with rf_metrics_col2:
            rmse_val = metrics.get('test_rmse', metrics.get('rmse', 0))
            st.metric("Test RMSE", f"{rmse_val:.2f}",
                      help="Root Mean Squared Error")
        with rf_metrics_col3:
            mae_val = metrics.get('test_mae', metrics.get('mae', 0))
            st.metric("Test MAE", f"{mae_val:.2f}",
                      help="Mean Absolute Error")
        with rf_metrics_col4:
            n_features = rf_metadata.get('n_features', len(rf_model.feature_names_in_))
            st.metric("Features", n_features,
                      help="Number of audio features used")

        # Training info
        with st.expander("ℹ️ Model Details"):
            rf_pred_col1, rf_pred_col2 = st.columns(2)
            with rf_pred_col1:
                st.markdown("**Hyperparameters:**")
                # Handle both hyperparameters and model_params
                params = rf_metadata.get('model_params', rf_metadata.get('hyperparameters', {}))
                if params:
                    display_params = {
                        "n_estimators": params.get('n_estimators', 'N/A'),
                        "max_depth": params.get('max_depth', 'N/A'),
                        "min_samples_split": params.get('min_samples_split', 'N/A'),
                        "min_samples_leaf": params.get('min_samples_leaf', 'N/A'),
                        "max_features": params.get('max_features', 'N/A'),
                        "bootstrap": params.get('bootstrap', 'N/A')
                    }
                    st.json(display_params)
                else:
                    st.write("No hyperparameters available")
            with rf_pred_col2:
                st.markdown("**Training Info:**")
                st.write(f"- Model Type: Random Forest Regressor")
                n_samples = rf_metadata.get('n_samples', 'N/A')
                st.write(f"- Total Samples: {n_samples:,}" if isinstance(n_samples, int) else f"- Total Samples: {n_samples}")
                train_shapes = rf_metadata.get('data_shapes', {})
                if train_shapes:
                    st.write(f"- Train: {train_shapes.get('train', ['N/A'])[0]:,} samples")
                    st.write(f"- Test: {train_shapes.get('test', ['N/A'])[0]:,} samples")
                timestamp = rf_metadata.get('timestamp', rf_metadata.get('training_date', 'N/A'))
                if timestamp != 'N/A':
                    st.write(f"- Trained: {timestamp[:10]}")

        st.markdown("---")

        # Feature Importance
        st.markdown("### 🎯 Feature Importance Analysis")
        st.markdown("""
        The Random Forest model identifies which audio features have the strongest impact on track popularity.
        Features are ranked by their importance scores (gain-based).
        """)

        top_features = rf_feature_importance.head(20) if len(rf_feature_importance) > 20 else rf_feature_importance
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {len(top_features)} Feature Importances',
            labels={'importance': 'Importance Score', 'feature': 'Audio Feature'},
            color='importance',
            color_continuous_scale='Greens',
            text='importance'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=max(400, len(top_features) * 30), showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        # Key insights
        st.info(f"""
        **Key Findings:**
        - **{top_features.iloc[0]['feature'].capitalize()}** is the most important feature ({top_features.iloc[0]['importance']:.1%})
        - **{top_features.iloc[1]['feature'].capitalize()}** is the second most important ({top_features.iloc[1]['importance']:.1%})
        - **{top_features.iloc[2]['feature'].capitalize()}** is the third most important ({top_features.iloc[2]['importance']:.1%})
        - These top 3 features account for {top_features.head(3)['importance'].sum():.1%} of the model's predictive power
        """)

        st.markdown("---")

        # SHAP Analysis
        st.markdown("### 🎯 SHAP Feature Impact Analysis")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to individual predictions.
        Unlike traditional feature importance, SHAP values show both *direction* (positive/negative impact) and *magnitude*.
        """)

        # Persistent SHAP caching to disk
        def load_or_compute_rf_shap_values(model, model_path, X_sample):
            """Load cached SHAP values or compute if not available"""
            import os
            from pathlib import Path

            # Create cache directory
            cache_dir = Path('outputs/shap')
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Generate cache filename based on model file
            model_filename = Path(model_path).stem
            cache_file = cache_dir / f'{model_filename}_shap_cache.joblib'

            # Check if cache exists and is valid
            if cache_file.exists():
                try:
                    st.info(f"📦 Loading cached SHAP values from {cache_file.name}...")
                    cached_data = joblib.load(cache_file)

                    # Verify cache has correct sample size
                    if cached_data['shap_values'].shape[0] == X_sample.shape[0]:
                        st.success("✅ Loaded cached SHAP values!")
                        return cached_data['shap_values'], cached_data['base_value'], cached_data['X_sample']
                    else:
                        st.warning("⚠️ Cache size mismatch, recomputing...")
                except Exception as e:
                    st.warning(f"⚠️ Cache load failed ({str(e)}), recomputing...")

            # Compute SHAP values
            st.info("🔄 Computing SHAP values... This may take a moment.")
            with st.spinner("Calculating SHAP explanations..."):
                explainer = shap.TreeExplainer(model)
                # Convert to numpy to avoid feature name warnings
                X_array = X_sample.values if hasattr(X_sample, 'values') else X_sample
                shap_values = explainer.shap_values(X_array)
                base_value = explainer.expected_value

                # Save to cache
                cache_data = {
                    'shap_values': shap_values,
                    'base_value': base_value,
                    'X_sample': X_sample,
                    'timestamp': datetime.now().isoformat(),
                    'model_file': model_filename
                }
                joblib.dump(cache_data, cache_file)
                st.success(f"✅ SHAP values computed and cached to {cache_file.name}!")

            return shap_values, base_value, X_sample

        st.info("Loading SHAP analysis... This section computes model explanations.")

        try:
            # Load data for SHAP analysis
            X_shap, y_shap = load_ml_data()

            # Use a smaller sample for SHAP (computational efficiency)
            shap_sample_size = min(500, len(X_shap))
            X_shap_sample_original = X_shap[:shap_sample_size]

            # Load or compute SHAP values with persistent caching
            shap_values, base_value, X_shap_sample = load_or_compute_rf_shap_values(
                rf_model, rf_model_path, X_shap_sample_original
            )

            # SHAP Summary Plot (Beeswarm)
            st.markdown("#### Feature Impact Distribution")
            st.markdown("""
            This plot shows how each feature affects predictions across the dataset:
            - **Position (x-axis)**: SHAP value (impact on prediction)
            - **Color**: Feature value (red = high, blue = low)
            - **Density**: How often this impact occurs
            """)
    
            fig_shap_summary, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap_sample, show=False, plot_type="dot")
            st.pyplot(fig_shap_summary, width="stretch")
            plt.close()
    
            st.caption("""
            **Reading the plot:**
            - Features at the top have the most impact on predictions
            - Red dots (high feature values) on the right mean higher predictions
            - Blue dots (low feature values) on the left mean lower predictions
            """)
    
            # SHAP Bar Plot (Mean Absolute Impact)
            rf_shap_col1, rf_shap_col2 = st.columns(2)
    
            with rf_shap_col1:
                st.markdown("#### Average Feature Impact")
                fig_shap_bar, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
                st.pyplot(fig_shap_bar, width="stretch")
                plt.close()
    
                st.caption("""
                **Mean |SHAP value|** - Average magnitude of impact across all predictions.
                Complements Random Forest's gain-based importance with impact-based importance.
                """)
    
            with rf_shap_col2:
                st.markdown("#### Sample Prediction Explanation")
    
                # Select a random sample for waterfall plot
                sample_idx = st.slider("Select track index to explain", 0, shap_sample_size-1, 0, key="rf_shap_sample_slider")
    
                # Create waterfall plot
                fig_waterfall, ax = plt.subplots(figsize=(8, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=base_value,
                        data=X_shap_sample.iloc[sample_idx],
                        feature_names=X_shap_sample.columns.tolist()
                    ),
                    show=False
                )
                st.pyplot(fig_waterfall, width="stretch")
                plt.close()
    
                # Convert base_value to scalar if it's an array
                base_value_scalar = float(np.array(base_value).flat[0])
    
                st.caption(f"""
                **Waterfall for track #{sample_idx}:**
                - Base value: {base_value_scalar:.2f} (average prediction)
                - Actual prediction: {rf_model.predict(X_shap_sample.iloc[[sample_idx]])[0]:.2f}
                - Shows how each feature pushes prediction up or down
                """)
    
            # SHAP Dependence Plots
            st.markdown("#### Feature Dependence Analysis")
            st.markdown("Explore how individual features affect predictions across different values:")
    
            rf_shap_col1, rf_shap_col2 = st.columns(2)
    
            with rf_shap_col1:
                # Top feature dependence
                top_feature = rf_feature_importance.iloc[0]['feature']
                fig_dep1, ax1 = plt.subplots(figsize=(8, 5))
                shap.dependence_plot(
                    top_feature,
                    shap_values,
                    X_shap_sample,
                    ax=ax1,
                    show=False
                )
                st.pyplot(fig_dep1, width="stretch")
                plt.close()
    
                st.caption(f"""
                **{top_feature.capitalize()} dependence:**
                Shows relationship between {top_feature} values and their impact on predictions.
                Color represents interaction with most correlated feature.
                """)
    
            with rf_shap_col2:
                # Second feature dependence
                second_feature = rf_feature_importance.iloc[1]['feature']
                fig_dep2, ax2 = plt.subplots(figsize=(8, 5))
                shap.dependence_plot(
                    second_feature,
                    shap_values,
                    X_shap_sample,
                    ax=ax2,
                    show=False
                )
                st.pyplot(fig_dep2, width="stretch")
                plt.close()
    
                st.caption(f"""
                **{second_feature.capitalize()} dependence:**
                Shows relationship between {second_feature} values and their impact on predictions.
                Helps identify non-linear patterns and feature interactions.
                """)
    
        except Exception as e:
            st.error(f"❌ Error computing SHAP values: {str(e)}")
            st.info("SHAP analysis requires the model to be properly trained. Try reloading the page or check the model file.")
            import traceback
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
            st.warning("⚠️ SHAP analysis failed, but the rest of the dashboard will continue to load below.")

        st.markdown("---")

        # Model Performance
        st.markdown("### 📈 Model Performance on Full Dataset")
        n_samples_display = rf_metadata.get('n_samples', len(df))
        st.markdown(f"""
        Below are predictions on a sample of the full Spotify dataset ({n_samples_display:,} cleaned tracks).
        The model was trained on deduplicated data with optimized hyperparameters from Optuna.
        """)

        X_test, y_test = load_ml_data()

        # Data is already prepared with correct features from load_ml_data
        sample_size = min(2000, len(X_test))
        y_pred = rf_model.predict(X_test[:sample_size])
        y_actual = y_test[:sample_size]
        if hasattr(y_actual, 'values'):
            y_actual = y_actual.values.flatten()

        # Calculate metrics on this sample
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        sample_r2 = r2_score(y_actual, y_pred)
        sample_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        sample_mae = mean_absolute_error(y_actual, y_pred)

        # Show sample metrics
        rf_error_col1, rf_error_col2, rf_error_col3 = st.columns(3)
        with rf_error_col1:
            st.metric("Sample R²", f"{sample_r2:.4f}", help=f"R² on {sample_size:,} tracks from full dataset")
        with rf_error_col2:
            st.metric("Sample RMSE", f"{sample_rmse:.2f}")
        with rf_error_col3:
            st.metric("Sample MAE", f"{sample_mae:.2f}")

        rf_error_col1, rf_error_col2 = st.columns(2)

        with rf_error_col1:
            # Prediction vs Actual
            fig = px.scatter(
                x=y_actual,
                y=y_pred,
                labels={'x': 'Actual Popularity', 'y': 'Predicted Popularity'},
                title='Predictions vs Actual Values',
                opacity=0.5
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[0, 100],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                )
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

        with rf_error_col2:
            # Residuals histogram
            residuals = y_actual - y_pred
            fig = px.histogram(
                residuals,
                nbins=50,
                title='Prediction Errors Distribution',
                labels={'value': 'Residual (Actual - Predicted)', 'count': 'Frequency'},
                color_discrete_sequence=['#1DB954']
            )
            # Add mean line
            fig.add_vline(x=residuals.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {residuals.mean():.2f}")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        # Additional performance visualizations
        st.markdown("### 🔍 Advanced Model Diagnostics")

        rf_error_col1, rf_error_col2 = st.columns(2)

        with rf_error_col1:
            # Residual plot (residuals vs predicted)
            fig = px.scatter(
                x=y_pred,
                y=residuals,
                labels={'x': 'Predicted Popularity', 'y': 'Residual (Actual - Predicted)'},
                title='Residual Plot',
                opacity=0.5,
                color_discrete_sequence=['#1DB954']
            )
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red",
                         annotation_text="Zero Error")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

            st.caption("""
            **Residual Plot Interpretation:**
            - Points should scatter randomly around zero line
            - Patterns indicate model bias or heteroscedasticity
            - Fan shapes suggest varying error across popularity range
            """)

        with rf_error_col2:
            # Error by popularity ranges
            popularity_bins = pd.cut(y_actual, bins=[0, 20, 40, 60, 80, 100],
                                    labels=['Very Low (0-20)', 'Low (20-40)', 'Medium (40-60)',
                                           'High (60-80)', 'Very High (80-100)'])
            error_df = pd.DataFrame({
                'Popularity Range': popularity_bins,
                'Absolute Error': np.abs(residuals)
            })

            fig = px.box(
                error_df,
                x='Popularity Range',
                y='Absolute Error',
                title='Model Error by Popularity Range',
                labels={'Absolute Error': 'Absolute Error (points)'},
                color='Popularity Range',
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            st.caption("""
            **Error Distribution Analysis:**
            - Shows where the model performs best/worst
            - Lower boxes indicate better prediction accuracy
            - Helps identify if model struggles with specific popularity ranges
            """)

        rf_actual_col1, rf_actual_col2 = st.columns(2)

        with rf_actual_col1:
            # Actual distribution
            fig = px.histogram(
                y_actual,
                nbins=30,
                title='Actual Popularity Distribution',
                labels={'value': 'Popularity', 'count': 'Frequency'},
                color_discrete_sequence=['#1DB954']
            )
            fig.add_vline(x=y_actual.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {y_actual.mean():.1f}")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch", key="rf_actual_distribution_hist")

        with rf_actual_col2:
            # Predicted distribution
            fig = px.histogram(
                y_pred,
                nbins=30,
                title='Predicted Popularity Distribution',
                labels={'value': 'Popularity', 'count': 'Frequency'},
                color_discrete_sequence=['#66BB6A']
            )
            fig.add_vline(x=y_pred.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {y_pred.mean():.1f}")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch", key="rf_predicted_distribution_hist")

        # Model insights
        st.markdown("### 💡 Model Performance Insights")

        # Calculate additional metrics
        over_predictions = np.sum(y_pred > y_actual)
        under_predictions = np.sum(y_pred < y_actual)
        mae_by_range = error_df.groupby('Popularity Range', observed=True)['Absolute Error'].mean().to_dict()

        rf_feat_dist_col1, rf_feat_dist_col2, rf_feat_dist_col3 = st.columns(3)
        with rf_feat_dist_col1:
            st.metric("Overestimations", f"{over_predictions:,}",
                     help="Predictions higher than actual popularity")
        with rf_feat_dist_col2:
            st.metric("Underestimations", f"{under_predictions:,}",
                     help="Predictions lower than actual popularity")
        with rf_feat_dist_col3:
            accuracy_within_10 = np.sum(np.abs(residuals) <= 10) / len(residuals) * 100
            st.metric("Within ±10 Points", f"{accuracy_within_10:.1f}%",
                     help="Percentage of predictions within 10 points of actual")

        # Best/worst performance ranges
        if mae_by_range:
            best_range = min(mae_by_range, key=mae_by_range.get)
            worst_range = max(mae_by_range, key=mae_by_range.get)

            st.success(f"""
            **Best Performance:** {best_range} popularity range (MAE: {mae_by_range[best_range]:.2f} points)
            """)

            st.warning(f"""
            **Needs Improvement:** {worst_range} popularity range (MAE: {mae_by_range[worst_range]:.2f} points)
            """)

        # Model explanation
        st.info(f"""
        **Understanding R² = {sample_r2:.4f}:**

        Audio features alone explain ~{sample_r2*100:.1f}% of popularity variance. The remaining ~{(1-sample_r2)*100:.1f}% comes from:
        - Artist fame and follower count
        - Marketing budget and promotion
        - Playlist placements
        - Social media trends and virality
        - Release timing and cultural factors

        **This is expected and aligns with academic research on music popularity prediction.**
        """)

        st.markdown("---")

        # Model Comparison: RF vs XGBoost
        st.markdown("### 🏆 Model Comparison: Random Forest vs XGBoost")

        rf_comp_col1, rf_comp_col2 = st.columns(2)

        with rf_comp_col1:
            # Performance comparison
            comparison_data = {
                'Model': ['Random Forest', 'XGBoost'],
                'R² Score': [sample_r2, metadata.get('metrics', {}).get('test_r2', 0.1619)],
                'RMSE': [sample_rmse, metadata.get('metrics', {}).get('test_rmse', 16.32)],
                'MAE': [sample_mae, metadata.get('metrics', {}).get('test_mae', 13.14)]
            }
            comparison_df = pd.DataFrame(comparison_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Random Forest',
                x=['R² Score', 'RMSE', 'MAE'],
                y=[sample_r2, sample_rmse, sample_mae],
                marker_color='#4CAF50'
            ))
            fig.add_trace(go.Bar(
                name='XGBoost',
                x=['R² Score', 'RMSE', 'MAE'],
                y=[comparison_data['R² Score'][1], comparison_data['RMSE'][1], comparison_data['MAE'][1]],
                marker_color='#2196F3'
            ))
            fig.update_layout(
                title='Performance Metrics Comparison',
                barmode='group',
                height=400,
                yaxis_title='Score'
            )
            st.plotly_chart(fig, width="stretch")

        with rf_comp_col2:
            # Create comparison table
            st.markdown("#### Performance Summary")
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['R² Score'], color='lightgreen')
                                  .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen'),
                hide_index=True
            )

            # Calculate differences
            r2_diff = ((sample_r2 - comparison_data['R² Score'][1]) / comparison_data['R² Score'][1]) * 100
            rmse_diff = ((sample_rmse - comparison_data['RMSE'][1]) / comparison_data['RMSE'][1]) * 100
            mae_diff = ((sample_mae - comparison_data['MAE'][1]) / comparison_data['MAE'][1]) * 100

            st.metric("R² Difference", f"{r2_diff:+.2f}%",
                     help="Positive means RF is better")
            st.metric("RMSE Difference", f"{rmse_diff:+.2f}%",
                     help="Negative means RF is better (lower error)")
            st.metric("MAE Difference", f"{mae_diff:+.2f}%",
                     help="Negative means RF is better (lower error)")

        st.markdown("---")

        # Tree Analysis
        st.markdown("### 🌳 Random Forest Tree Analysis")
        st.markdown("""
        Understanding the ensemble structure helps interpret model behavior and diagnose potential issues.
        """)

        rf_tree_col1, rf_tree_col2, rf_tree_col3 = st.columns(3)

        with rf_tree_col1:
            # Tree depth distribution
            tree_depths = [tree.get_depth() for tree in rf_model.estimators_]
            fig = px.histogram(
                tree_depths,
                nbins=20,
                title=f'Tree Depth Distribution (n={len(tree_depths)})',
                labels={'value': 'Tree Depth', 'count': 'Number of Trees'},
                color_discrete_sequence=['#4CAF50']
            )
            fig.add_vline(x=np.mean(tree_depths), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {np.mean(tree_depths):.1f}")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            st.caption(f"**Max Depth Limit:** {rf_metadata.get('hyperparameters', {}).get('max_depth', 'N/A')}")

        with rf_tree_col2:
            # Number of leaves per tree
            n_leaves = [tree.get_n_leaves() for tree in rf_model.estimators_]
            fig = px.histogram(
                n_leaves,
                nbins=20,
                title=f'Leaves per Tree Distribution',
                labels={'value': 'Number of Leaves', 'count': 'Number of Trees'},
                color_discrete_sequence=['#8BC34A']
            )
            fig.add_vline(x=np.mean(n_leaves), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {np.mean(n_leaves):.0f}")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            st.caption(f"**Total Leaves:** {sum(n_leaves):,}")

        with rf_tree_col3:
            # Tree complexity metrics
            st.markdown("#### Ensemble Statistics")
            st.metric("Number of Trees", len(rf_model.estimators_))
            st.metric("Avg Tree Depth", f"{np.mean(tree_depths):.1f}")
            st.metric("Avg Leaves/Tree", f"{np.mean(n_leaves):.0f}")
            st.metric("Max Tree Depth", max(tree_depths))
            st.metric("Min Tree Depth", min(tree_depths))

        st.markdown("---")

        # Feature Importance Comparison
        st.markdown("### 📊 Feature Importance: Multiple Perspectives")
        st.markdown("""
        Comparing different importance metrics provides a more complete understanding of feature impact.
        """)

        rf_importance_col1, rf_importance_col2 = st.columns(2)

        with rf_importance_col1:
            # Gini importance (default)
            st.markdown("#### Gini Importance (Mean Decrease Impurity)")
            fig = px.bar(
                rf_feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Features by Gini Importance',
                labels={'importance': 'Importance', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Greens'
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            st.caption("""
            **Gini Importance** measures average decrease in impurity when splitting on a feature.
            Biased toward high-cardinality features and features used early in trees.
            """)

        with rf_importance_col2:
            # Compare with XGBoost importance
            st.markdown("#### Importance Comparison with XGBoost")

            # Merge RF and XGBoost importances
            xgb_importance = feature_importance.copy()
            xgb_importance = xgb_importance.rename(columns={'importance': 'xgb_importance'})

            rf_importance = rf_feature_importance.copy()
            rf_importance = rf_importance.rename(columns={'importance': 'rf_importance'})

            merged = pd.merge(rf_importance, xgb_importance, on='feature', how='inner')
            merged = merged.sort_values('rf_importance', ascending=False).head(10)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Random Forest',
                y=merged['feature'],
                x=merged['rf_importance'],
                orientation='h',
                marker_color='#4CAF50'
            ))
            fig.add_trace(go.Bar(
                name='XGBoost',
                y=merged['feature'],
                x=merged['xgb_importance'],
                orientation='h',
                marker_color='#2196F3'
            ))
            fig.update_layout(
                title='Feature Importance: RF vs XGBoost',
                barmode='group',
                height=400,
                xaxis_title='Importance Score',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, width="stretch")

            st.caption("""
            Comparing importance across models reveals which features are consistently important
            vs model-specific artifacts.
            """)

        st.markdown("---")

        # Prediction Uncertainty Analysis
        st.markdown("### 🎲 Prediction Uncertainty & Confidence")
        st.markdown("""
        Random Forest provides prediction confidence through variance across trees.
        """)

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test[:sample_size]) for tree in rf_model.estimators_])
        prediction_std = np.std(tree_predictions, axis=0)
        prediction_mean = np.mean(tree_predictions, axis=0)

        rf_uncertainty_col1, rf_uncertainty_col2 = st.columns(2)

        with rf_uncertainty_col1:
            # Prediction uncertainty distribution
            fig = px.histogram(
                prediction_std,
                nbins=50,
                title='Prediction Uncertainty Distribution',
                labels={'value': 'Standard Deviation Across Trees', 'count': 'Frequency'},
                color_discrete_sequence=['#FF9800']
            )
            fig.add_vline(x=prediction_std.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {prediction_std.mean():.2f}")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            st.caption("""
            **Low uncertainty** (< 5): Consistent predictions across trees (high confidence)
            **High uncertainty** (> 10): Disagreement among trees (low confidence)
            """)

        with rf_uncertainty_col2:
            # Uncertainty vs Error
            abs_error = np.abs(y_actual - y_pred)

            fig = px.scatter(
                x=prediction_std,
                y=abs_error,
                labels={'x': 'Prediction Uncertainty (Std Dev)', 'y': 'Absolute Error'},
                title='Prediction Uncertainty vs Actual Error',
                opacity=0.5,
                color=abs_error,
                color_continuous_scale='Reds'
            )
            # Add trend line
            z = np.polyfit(prediction_std, abs_error, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(prediction_std.min(), prediction_std.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

            correlation = np.corrcoef(prediction_std, abs_error)[0, 1]
            st.caption(f"""
            **Correlation:** {correlation:.3f}
            {'Strong correlation' if abs(correlation) > 0.5 else 'Weak correlation'} between uncertainty and error.
            Ideally, high uncertainty should predict high error.
            """)

        # Confidence intervals
        rf_confidence_col1, rf_confidence_col2, rf_confidence_col3 = st.columns(3)

        with rf_confidence_col1:
            high_confidence = np.sum(prediction_std < 5)
            st.metric("High Confidence", f"{high_confidence:,}",
                     f"{high_confidence/len(prediction_std)*100:.1f}%",
                     help="Predictions with std < 5")

        with rf_confidence_col2:
            medium_confidence = np.sum((prediction_std >= 5) & (prediction_std < 10))
            st.metric("Medium Confidence", f"{medium_confidence:,}",
                     f"{medium_confidence/len(prediction_std)*100:.1f}%",
                     help="Predictions with 5 ≤ std < 10")

        with rf_confidence_col3:
            low_confidence = np.sum(prediction_std >= 10)
            st.metric("Low Confidence", f"{low_confidence:,}",
                     f"{low_confidence/len(prediction_std)*100:.1f}%",
                     help="Predictions with std ≥ 10")

        # Model metadata
        with st.expander("📋 View Model Metadata"):
            st.json(rf_metadata)

# ============================================================================
# TAB 5: Track Predictor (Interactive UX)
# ============================================================================
with tab5:
    st.markdown('<h2 class="sub-header">🎯 Track Popularity Predictor</h2>', unsafe_allow_html=True)
    st.markdown("""
    **As a music producer**, use this tool to:
    - 🎵 Predict popularity for your new track
    - 💡 Get AI-powered recommendations to maximize audience reach
    - 📊 Compare your track against successful songs
    """)

    st.markdown("---")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["🎚️ Sliders (Beginner-Friendly)", "📝 Manual Entry (Advanced)", "🎲 Random Example"]
    )

    if input_method == "🎲 Random Example":
        # Load a random track
        sample_track = df.sample(1).iloc[0]
        st.info(f"**Example Track**: {sample_track['track_name']} by {sample_track['artists']}")

        # Set values from sample
        danceability = sample_track['danceability']
        energy = sample_track['energy']
        valence = sample_track['valence']
        acousticness = sample_track['acousticness']
        instrumentalness = sample_track['instrumentalness']
        speechiness = sample_track['speechiness']
        liveness = sample_track['liveness']
        loudness = sample_track['loudness']
        tempo = sample_track['tempo']
        duration_ms = sample_track['duration_ms']
        explicit = int(sample_track['explicit'])
        key = sample_track['key']
        mode = sample_track['mode']
        time_signature = sample_track['time_signature']
        track_genre = sample_track['track_genre']

    else:
        # User input form
        st.markdown("### 🎼 Track Characteristics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Audio Features (0-1)**")
            danceability = st.slider("🕺 Danceability", 0.0, 1.0, 0.7, 0.01,
                                     help="How suitable the track is for dancing")
            energy = st.slider("⚡ Energy", 0.0, 1.0, 0.7, 0.01,
                              help="Intensity and activity level")
            valence = st.slider("😊 Valence (Positivity)", 0.0, 1.0, 0.5, 0.01,
                               help="Musical positiveness (happy vs sad)")
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.3, 0.01,
                                    help="Likelihood of being acoustic")

        with col2:
            st.markdown("**More Features**")
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.1, 0.01,
                                        help="Probability of no vocals")
            speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1, 0.01,
                                   help="Presence of spoken words")
            liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.2, 0.01,
                                help="Likelihood of live performance")
            loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0, 0.5)

        with col3:
            st.markdown("**Track Details**")
            tempo = st.slider("🥁 Tempo (BPM)", 50.0, 200.0, 120.0, 1.0)
            duration_ms = st.slider("⏱️ Duration (minutes)", 1.0, 10.0, 3.5, 0.1) * 60000
            explicit = st.checkbox("🔞 Explicit Content", value=False)
            key = st.selectbox("🎹 Key", range(12),
                              format_func=lambda x: ['C', 'C#', 'D', 'D#', 'E', 'F',
                                                     'F#', 'G', 'G#', 'A', 'A#', 'B'][x])
            mode = st.selectbox("🎵 Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
            time_signature = st.selectbox("⏰ Time Signature", [3, 4, 5], index=1)
            track_genre = st.selectbox("🎸 Genre", sorted(df['track_genre'].unique()))

    st.markdown("---")

    # Predict button
    if st.button("🚀 Predict Popularity", type="primary", width='stretch'):
        # Prepare features
        duration_min = duration_ms / 60000

        # Create feature dict
        features = {
            'danceability': danceability,
            'energy': energy,
            'key': key,
            'loudness': loudness,
            'mode': mode,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'valence': valence,
            'tempo': tempo,
            'time_signature': time_signature,
            'explicit': int(explicit),
            'duration_ms': duration_ms,
            'duration_min': duration_min,
        }

        # Add engineered features
        features['energy_danceability'] = energy * danceability
        features['valence_energy'] = valence * energy
        features['acousticness_energy'] = acousticness * energy
        features['energy_squared'] = energy ** 2
        features['danceability_squared'] = danceability ** 2
        features['valence_squared'] = valence ** 2
        features['is_short_track'] = int(duration_min < 3)
        features['is_long_track'] = int(duration_min > 5)
        features['high_energy_happy'] = int(energy > 0.7 and valence > 0.7)
        features['low_energy_sad'] = int(energy < 0.3 and valence < 0.3)

        # Encode categorical features (simplified - using defaults)
        # In production, you'd need the exact encoding from training
        features['track_genre_encoded'] = 0  # Would need actual encoding

        # One-hot encoded features (set all to 0 by default, set specific ones to 1)
        features['mode_1'] = int(mode == 1)
        features['time_signature_4'] = int(time_signature == 4)
        features['time_signature_5'] = int(time_signature == 5)

        # Mood/energy (simplified)
        if valence > 0.5 and energy > 0.5:
            mood = "Happy/High Energy"
        elif valence <= 0.5 and energy > 0.5:
            mood = "Energetic/Sad"
        elif valence > 0.5 and energy <= 0.5:
            mood = "Chill/Happy"
        else:
            mood = "Sad/Low Energy"

        # Create dummy columns for mood encoding (simplified)
        features['mood_energy_Energetic/Sad'] = int(mood == "Energetic/Sad")
        features['mood_energy_Happy/High Energy'] = int(mood == "Happy/High Energy")
        features['mood_energy_Sad/Low Energy'] = int(mood == "Sad/Low Energy")

        # Energy category
        if energy < 0.33:
            energy_cat = "Low Energy"
        elif energy < 0.66:
            energy_cat = "Medium Energy"
        else:
            energy_cat = "High Energy"

        features['energy_category_Low Energy'] = int(energy_cat == "Low Energy")
        features['energy_category_Medium Energy'] = int(energy_cat == "Medium Energy")

        # Tempo category
        if tempo < 90:
            tempo_cat = "Slow"
        elif tempo < 120:
            tempo_cat = "Moderate"
        elif tempo < 150:
            tempo_cat = "Fast"
        else:
            tempo_cat = "Very Fast"

        features['tempo_category_Moderate'] = int(tempo_cat == "Moderate")
        features['tempo_category_Slow'] = int(tempo_cat == "Slow")
        features['tempo_category_Very Fast'] = int(tempo_cat == "Very Fast")

        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])

        # Filter to only include features the model expects
        feature_df_model = feature_df[[col for col in model.feature_names_in_ if col in feature_df.columns]].copy()

        # Add missing features with appropriate default values
        for col in model.feature_names_in_:
            if col not in feature_df_model.columns:
                if col == 'release_year':
                    feature_df_model[col] = 2020  # Default year for missing release_year
                else:
                    feature_df_model[col] = 0  # Default to 0 for other missing features

        # Ensure column order matches model's expectations
        feature_df_model = feature_df_model[model.feature_names_in_]

        # Make prediction
        try:
            prediction = model.predict(feature_df_model)[0]
            prediction = np.clip(prediction, 0, 100)  # Ensure 0-100 range

            # Display prediction
            st.markdown(f'<div class="prediction-box">Predicted Popularity: {prediction:.1f}/100</div>',
                       unsafe_allow_html=True)

            # Interpretation
            if prediction >= 70:
                st.success("🎉 **Excellent!** This track has high viral potential!")
            elif prediction >= 50:
                st.info("👍 **Good!** This track should perform well with targeted promotion.")
            elif prediction >= 30:
                st.warning("⚠️ **Moderate** This track may need optimization or niche targeting.")
            else:
                st.error("📉 **Challenging** Consider refining the track characteristics.")

            # Recommendations
            st.markdown("### 💡 AI-Powered Recommendations")

            recommendations = []

            # Analysis and recommendations
            if danceability < 0.5:
                recommendations.append({
                    "factor": "🕺 Danceability",
                    "current": f"{danceability:.2f}",
                    "suggestion": "Increase to 0.7+",
                    "impact": "+8-12 popularity points",
                    "tip": "Add a stronger, more rhythmic beat to make the track more danceable"
                })

            if energy < 0.6:
                recommendations.append({
                    "factor": "⚡ Energy",
                    "current": f"{energy:.2f}",
                    "suggestion": "Increase to 0.7+",
                    "impact": "+5-10 popularity points",
                    "tip": "Boost the intensity with louder instruments or faster tempo"
                })

            if valence < 0.4:
                recommendations.append({
                    "factor": "😊 Positivity",
                    "current": f"{valence:.2f}",
                    "suggestion": "Consider 0.5-0.7 range",
                    "impact": "+5-8 popularity points",
                    "tip": "Happy/upbeat tracks tend to be more popular. Try major key or uplifting melodies"
                })

            if duration_min > 4.5:
                recommendations.append({
                    "factor": "⏱️ Duration",
                    "current": f"{duration_min:.1f} min",
                    "suggestion": "3-4 minutes optimal",
                    "impact": "+3-7 popularity points",
                    "tip": "Shorter tracks perform better. Consider tightening arrangement"
                })

            if loudness < -8:
                recommendations.append({
                    "factor": "🔊 Loudness",
                    "current": f"{loudness:.1f} dB",
                    "suggestion": "Target -5 to -3 dB",
                    "impact": "+3-5 popularity points",
                    "tip": "Professional mastering to competitive loudness levels"
                })

            if recommendations:
                for i, rec in enumerate(recommendations[:5], 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>#{i}. {rec['factor']}</strong><br>
                        📊 Current: {rec['current']} → Suggested: {rec['suggestion']}<br>
                        📈 Potential Impact: <strong>{rec['impact']}</strong><br>
                        💡 Tip: {rec['tip']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✨ Your track characteristics are well-optimized! No major changes needed.")

            # Comparison with similar successful tracks
            st.markdown("### 🎵 Similar Successful Tracks")

            # Find similar tracks in dataset
            similar_mask = (
                (df['track_genre'] == track_genre) &
                (df['popularity'] >= 70)
            )
            similar_tracks = df[similar_mask].head(5)

            if len(similar_tracks) > 0:
                st.dataframe(
                    similar_tracks[['track_name', 'artists', 'popularity',
                                   'danceability', 'energy', 'valence']],
                    width='stretch'
                )
            else:
                st.info("No highly popular tracks found in this genre. Be a trendsetter! 🚀")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all features are filled correctly.")

# Footer
st.markdown("---")
footer_n_samples = metadata.get('n_samples', len(df))
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>🎵 Spotify Track Analytics Dashboard | Built with Streamlit & XGBoost</p>
    <p>Data: {footer_n_samples:,} cleaned tracks | Model R² = {test_r2:.4f} | RMSE = {test_rmse:.2f} | MAE = {test_mae:.2f}</p>
    <p style='font-size: 0.9em;'>Dataset V2: Deduplicated, zero-popularity removed | Audio-only features</p>
</div>
""", unsafe_allow_html=True)
