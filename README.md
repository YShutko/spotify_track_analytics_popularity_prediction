# Spotify Track Analytics Popularity Prediction

# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

<p align="center">
  <img src="assets/teamlogo.png" alt="Project Logo" width="25%">
</p>


## Content
* [Readme.md](https://github.com/YShutko/spotify_track_analytics_popularity_prediction/blob/1eb084f166f61e2ec0c6dcf23cdb3fea6f7f3cb8/README.md)
* [Kanban Project Board](https://github.com/users/YShutko/projects/6)
* [Datasets](https://github.com/YShutko/spotify_track_analytics_popularity_prediction/tree/926c78849c653732de5f592223b0ff7cda01fab8/data)
* [Interactive Dashboard](app.py) - Streamlit web application
* [Jupyter Notebooks](notebooks/) - ETL, Feature Engineering, ML Model
* [ML Pipeline Summary](docs/ML_PIPELINE_SUMMARY.md) - Complete pipeline documentation

## 🚀 Quick Start

### 🌐 Try the Live Demo

**No installation required!** Try the interactive dashboard now:

**[🎵 Launch Spotify Track Analytics on Hugging Face Spaces](https://huggingface.co/spaces/jmbarlow/spotify-track-analytics)**

### Run Locally

**Option 1: Streamlit (recommended)**
```bash
# Using make command
make dashboard

# Or directly
streamlit run app.py
```
Opens at http://localhost:8501

**Option 2: Gradio**
```bash
# Using make command
make dashboard-gradio

# Or directly
python app_gradio.py
```
Opens at http://localhost:7860

All dashboards provide:
- 📊 **Data Explorer**: Browse and filter 114,000 tracks
- 📈 **Rich Visualizations**: Interactive charts and insights
- 🤖 **ML Model Analysis**: Feature importance and performance
- 🎯 **Track Predictor**: Predict popularity + get AI recommendations

### Run the ML Pipeline
```bash
# Complete ETL, feature engineering, and model training
python src/etl_pipeline.py
python src/feature_engineering.py
python src/train_model.py
```

## Dataset Content
The data set used for this project: [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). The collection of ~114,000 songs across 125 genres with features like danceability, energy, tempo, and popularity. Ideal for audio analysis, genre classification, and music trend exploration.

The dataset consists of the following columns:
* track_id: Unique Spotify identifier for each track.
* artists: List of artists performing the track, separated by semicolons.
* album_name: Title of the album where the track appears.
* track_name: Title of the song.
* popularity: Score from 0–100 based on recent play counts; higher means more popular.
* duration_ms: Length of the track in milliseconds.
* explicit: Indicates whether the track contains explicit content (True/False).
* danceability: Score (0.0–1.0) measuring how suitable the song is for dancing.
* energy: Score (0.0–1.0) reflecting intensity, speed, and loudness.
* key: Musical key using Pitch Class notation (0 = C, 1 = C♯/D♭, etc.).
* loudness: Overall volume of the track in decibels.
* mode: Indicates scale type (1 = major, 0 = minor).
* speechiness: Score estimating spoken content in the track.
* cousticness: Likelihood (0.0–1.0) that the song is acoustic.
* instrumentalness: Probability that the track has no vocals.
* liveness: Measures if the song was recorded live (higher = more live).
* valence: Positivity of the music (0.0 = sad, 1.0 = happy).
* tempo: Speed of the song in beats per minute (BPM).
time_signature: Musical meter (e.g. 4 = 4/4 time).
track_genre: Musical genre classification of the track.

## Business Requirements & Solutions
1. **Understand Key Drivers of Song Popularity**
   Stakeholders want to know which track features (e.g., genre, energy, length) drive popularity.
**Solution**: XGBoost ML model with feature importance analysis
- Interactive dashboard shows top 20 most important features
- Genre is the strongest predictor (importance: 0.0986)
- Track length, explicit content, and energy levels are significant
- **Access**: ML Model tab in Streamlit dashboard
2. **Classify Songs by Mood and Energy**
  The product team requires a system to categorize songs by mood for mood-based playlists.
**Solution**: Automated mood/energy classification system
- 4 mood categories: Happy/High Energy, Energetic/Sad, Chill/Happy, Sad/Low Energy
- Based on valence (positivity) and energy scores
- Visualized in interactive pie charts
- **Access**: Visualizations tab → Mood & Energy section
3. **Genre-Level Analysis**
  Marketing and analytics teams need insights into genre trends and performance.
**Solution**: Comprehensive genre analytics and visualizations
- 114 unique genres analyzed across 114,000 tracks
- Top 20 genres by track count displayed
- Genre-specific popularity trends
- **Access**: Data Explorer + Visualizations tabs
4. **Support Playlist Curation**
  Users should be able to filter and find similar tracks for playlist creation.
**Solution**: Advanced filtering and similarity search
- Filter by genre, popularity range, and audio features
- Download curated track lists as CSV
- Find similar successful tracks in the same genre
- **Access**: Data Explorer tab with multi-select filters
5. **Data-Driven Music Recommendations**
  Music producers need predictive tools for optimizing new track performance.
**Solution**: AI-powered recommendation engine for track optimization
- Predict popularity for new tracks (0-100 score)
- Get up to 5 actionable recommendations to maximize audience
- See potential impact of each change (+5 to +12 popularity points)
- Compare against successful tracks in the same genre
- **Access**: Track Predictor tab (music producer tool)


## Hypothesis
1. Genre is a strong predictor of popularity.
 Confirmed by: XGBoost feature importance — Genre ranked #1 (importance score: 0.0986).
2 Songs with higher energy and valence are perceived as more positive/mood-lifting.
Confirmed by: Mood classification into 4 quadrants using energy and valence; shown in mood pie chart.
3. Explicit content and short track length tend to lower popularity.
Confirmed by: Negative correlation in model feature importance scores and popularity predictions.
4. Most popular genres have a wider spread of track counts and variability in popularity.
Confirmed by: Genre distribution charts and popularity histograms in the Genre Analysis dashboard.
5. Audio features can be used to accurately predict a track’s future popularity.
Confirmed by: Model predictions (0–100 scale) with ~80% accuracy and test-set performance.

## Project Plan
* Data Acquisition & Preparation
  * Load and explore the Spotify tracks dataset from Kaggle.
  * Clean and preprocess data: handle missing values, convert duration to minutes, normalize/scale relevant features.
* Exploratory Data Analysis (EDA)
  * Analyze distribution of popularity, tempo, valence, energy, and other audio features.
  * Visualize relationships between key features using:
      * Correlation heatmaps
      * Pairplots and histograms
* Interactive Streamlit Dashboard 
  * Full-featured web application with 4 tabs:
    * Data Explorer - Browse and filter tracks
    * Visualizations - Rich, interactive charts
    * ML Model - Feature importance and performance
    * Track Predictor - Predict + optimize new tracks
  * Real-time popularity predictions with AI recommendations
  * Music producer workflow for track optimization
    
## The rationale to map the business requirements to the Data Visualisations


## Dashboard Design
### Streamlit Web Application
Launch with: `streamlit run app.py`

Tab 1: Data Explorer 
* Browse 114,000 Spotify tracks
* Filter by genre and popularity range
* View track details (name, artists, audio features)
* Download filtered data as CSV

Tab 2: Visualizations 
* **Popularity Distribution**: Histogram of track popularity scores
* **Audio Features**: 6 feature distributions (danceability, energy, valence, etc.)
* **Top 20 Genres**: Bar chart of most popular genres
* **Mood & Energy**: Pie charts showing track classification
* **Correlation Heatmap**: Feature relationships
* **Interactive Scatter**: Custom X/Y axis selection with color coding

Tab 3: ML Model 
* **Model Performance Metrics**: R² = 0.39, RMSE = 17.4
* **Top 20 Feature Importances**: Genre, track length, explicit content
* **Prediction vs Actual**: Scatter plot visualization
* **Residual Analysis**: Error distribution
* **Complete Model Metadata**: JSON export of specifications

Tab 4: Track Predictor 🎯
For Music Producers: Interactive tool to predict and optimize track popularity
**Input Methods**:
  * Sliders (beginner-friendly with tooltips)
  * Manual entry (advanced users)
  * Random example (explore real tracks)
**Features**:
*  Set 14 audio characteristics (danceability, energy, tempo, etc.)
* Get instant popularity prediction (0-100 score)
* Receive up to 5 AI-powered recommendations
* See potential impact (+5 to +12 popularity points)
* Compare with successful tracks in same genre
**Example Recommendations**:
* "Increase danceability from 0.5 to 0.7 for +10 points"
* "Boost energy with louder instruments for +8 points"
* "Try major key for more uplifting feel (+6 points)"

See [Streamlit App Guide](docs/streamlit_app_guide.md) for detailed usage instructions.

## Analysis techniques used
* Visual Studio Code
* Python
* Jupyter notebook
* ChatGPT
* Claude

## Ethical consideration
This project utilizes publicly available Spotify track data for the purpose of educational data analysis and machine learning model development. The following ethical aspects were taken into account:
* **Data Privacy:**  
  The dataset does not include any personal or user-specific information. All data refers to music track features and metadata that are publicly accessible through Spotify APIs. No individual listeners are identified or targeted.
* **Copyright and Usage Rights:**  
  Although track metadata (e.g., names, artists, albums) is included, no copyrighted audio content is used. The dataset complies with fair use for academic and non-commercial research purposes.
* **Bias and Fairness:**  
  The dataset may reflect biases in genre popularity or artist representation due to Spotify’s algorithmic and user-driven metrics. These biases can affect model predictions and should be acknowledged in analysis or applications.
* **Responsible Use of Predictions:**  
  Any popularity or mood predictions made using this data should not be used to marginalize artists or genres. The results represent trends in the data, not objective quality judgments.
* **Transparency:**  
  All analysis steps, assumptions, and model performance evaluations are documented for transparency. This promotes responsible AI usage and allows users to understand how conclusions were derived.
  
## Development Roadmap

## Planning:
* GitHub [Project Board]([https://github.com/users/YShutko/projects/3](https://github.com/users/YShutko/projects/6)) was used to plan and track the progress.

## Interactive Dashboard Features

### 🎵 Streamlit Web Application
Launch with: `streamlit run app.py`

#### Tab 1: Data Explorer 📊
- Browse 114,000 Spotify tracks
- Filter by genre and popularity range
- View track details (name, artists, audio features)
- Download filtered data as CSV

#### Tab 2: Visualizations 📈
- **Popularity Distribution**: Histogram of track popularity scores
- **Audio Features**: 6 feature distributions (danceability, energy, valence, etc.)
- **Top 20 Genres**: Bar chart of most popular genres
- **Mood & Energy**: Pie charts showing track classification
- **Correlation Heatmap**: Feature relationships
- **Interactive Scatter**: Custom X/Y axis selection with color coding

#### Tab 3: ML Model 🤖
- **Model Performance Metrics**: R² = 0.39, RMSE = 17.4
- **Top 20 Feature Importances**: Genre, track length, explicit content
- **Prediction vs Actual**: Scatter plot visualization
- **Residual Analysis**: Error distribution
- **Complete Model Metadata**: JSON export of specifications

#### Tab 4: Track Predictor 🎯
**For Music Producers**: Interactive tool to predict and optimize track popularity

**Input Methods**:
- 🎚️ Sliders (beginner-friendly with tooltips)
- 📝 Manual entry (advanced users)
- 🎲 Random example (explore real tracks)

**Features**:
- Set 14 audio characteristics (danceability, energy, tempo, etc.)
- Get instant popularity prediction (0-100 score)
- Receive up to 5 AI-powered recommendations
- See potential impact (+5 to +12 popularity points)
- Compare with successful tracks in same genre

**Example Recommendations**:
- "Increase danceability from 0.5 to 0.7 for +10 points"
- "Boost energy with louder instruments for +8 points"
- "Try major key for more uplifting feel (+6 points)"

See [Streamlit App Guide](docs/streamlit_app_guide.md) for detailed usage instructions.

## Main Data Analysis & ML Libraries
* **Data Processing**: Pandas, Numpy
* **Visualization**: Plotly, Seaborn, Matplotlib
* **ML & Feature Engineering**: Scikit-learn, XGBoost, Feature-Engine
* **Web App**: Streamlit
* **Data Storage**: PyArrow (Parquet format)
 
## Credits 
* [The Code Institute](https://codeinstitute.net/) Learning Management System
* [VS Code](https://code.visualstudio.com/) was used to wite the code
* [ChatGPT](https://chatgpt.com/) was used to generate and debug code
* [README](https://github.com/Code-Institute-Solutions/da-README-template) template
* [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) data set was used for this project
* [gradio](https://www.gradio.app/)

# Acknowledgements
Thanks to our facilitator Emma Lamont and  tutors for their help and support. 
