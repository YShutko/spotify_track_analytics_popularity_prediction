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

## Quick Start

### Try the Live Demo

**No installation required!** Try the interactive dashboard now:

**[ Launch Spotify Track Analytics on Hugging Face Spaces](https://huggingface.co/spaces/jmbarlow/spotify-track-analytics)**

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
- **Data Explorer**: Browse and filter 114,000 tracks
- **Rich Visualizations**: Interactive charts and insights
- **ML Model Analysis**: Feature importance and performance
- **Track Predictor**: Predict popularity + get AI recommendations

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
5. Audio features can be used to predict a track’s future popularity accurately.
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
The visualizations used in this project were chosen to help uncover the relationships between audio features and track popularity, classify tracks by mood and energy, and provide intuitive insights for both technical and non-technical audiences.

1. Correlation Heatmap: Key Audio Features
- Image: key_feature_correlation_heatmap.png
  
This heatmap displays the correlation matrix for five key audio features: popularity, tempo, valence, energy, and danceability. The goal was to identify which features are interrelated. While valence and danceability show a moderate correlation (0.48), the weak correlation with popularity confirms the need for a more complex model like XGBoost to capture non-linear relationships.

2. Pairplot: Audio Features Distribution and Interaction
  - Image: key_features_pairplot.png
    
The pairplot complements the heatmap by visualizing feature distributions and pairwise relationships. Diagonal histograms reveal how each feature is distributed, while scatter plots between variables show clusters and potential trends. This guided the data preprocessing and feature selection phases.

3. Heatmap: Audio Features vs. Popularity
- Image: popularity_drivers_heatmap.png
  
This extended heatmap includes more audio features (loudness, acousticness, instrumentalness, etc.) and their correlation with popularity. While individual correlations with popularity are weak, this confirmed our hypothesis that popularity is influenced by a combination of features rather than a single metric.

4. Popularity by Mood & Energy Classification
- Image: popularity_vs_mood_energy.png
  
Tracks were categorized into 4 mood/energy clusters based on valence and energy. A violin plot was used to show popularity distribution per group. Interestingly, Happy/High Energy tracks show slightly higher popularity medians, supporting the business use case for mood-based playlist curation.

5. Popularity by Time Signature (Binned)
- Image: time_signature_binned_avg_popularity.png
  
This bar chart shows how time signature (a rhythmic element) correlates with popularity. While the difference is not drastic, tracks with a time signature below 4 (e.g., 3/4) have marginally higher average popularity, sparking potential insights for music production.

6. Top 20 Artists by Average Popularity
- Image: top_artists_popularity.png
  
This horizontal bar chart ranks the top 20 artists based on average popularity across tracks. It serves as a benchmark for aspiring artists and supports the project’s recommendation engine, which can compare new tracks against popular artists' features.

7. Distribution of Valence
- Image: distribution_valence.png
  
Understanding the distribution of valence (musical positivity) provides context for mood classification. The distribution is relatively symmetrical, suggesting a balanced dataset. This insight confirmed the feasibility of segmenting songs into mood categories for personalized recommendations.

8. Popularity by Danceability (Binned)
- Image: danceability_binned_avg_popularity.png
  
This bar chart visualizes average song popularity across binned danceability values (standardized). The pattern reveals that moderately danceable tracks—those with danceability around the normalized range of 0.08 to 0.88—tend to score higher in average popularity, peaking at 35.
In contrast, extremely low or high danceability appears to slightly reduce popularity. This insight challenges the common assumption that maximum danceability guarantees success and supports a data-driven balance in production. It contributes to the recommendation engine by identifying optimal danceability ranges for potential hit songs.

9. Distribution of Danceability
- Image: distribution_danceability.png
  
This histogram shows how the danceability feature is distributed across the dataset. The near-normal shape, centered around 0 (due to standardization), indicates a well-balanced feature with no extreme skew. This validates its use in modeling without further transformation. The even spread across low, medium, and high values also supports the use of binning for more granular analysis, as seen in rationale #8.

10. Distribution of Energy
- Image: distribution_energy.png
  
This histogram illustrates the distribution of the energy feature. Unlike danceability, energy shows a right-skewed distribution, with more songs having lower energy scores. This suggests that calmer tracks dominate the dataset. Identifying this skew helps guide preprocessing decisions and informs model tuning—especially when normalizing or using tree-based algorithms like XGBoost that are robust to skew.

11. Distribution of Popularity
- Image: distribution_popularity.png
  
The popularity distribution is heavily left-skewed, with the largest group of tracks having very low popularity scores. This confirms the long-tail nature of streaming data: a few tracks are extremely popular, while the majority receive little attention. This imbalance justifies using stratified sampling or resampling techniques when building predictive models and highlights the importance of focusing on relative rather than absolute popularity.

12. Distribution of Tempo
- Image: distribution_tempo.png
  
This histogram shows a multimodal distribution for tempo, reflecting the diversity of musical genres (e.g., ballads, EDM, hip-hop) within the dataset. Peaks around standardized 0 suggest a central tendency, but the multiple modes indicate that clustering or segmentation might benefit from using tempo as a feature. This also informs mood- or genre-specific recommendations.

13. Popularity by Acousticness (Binned)
- Image: acousticness_binned_avg_popularity.png
  
This bar chart compares average popularity across binned values of acousticness. A subtle U-shaped pattern emerges—songs with moderate acousticness tend to be more popular, while very high or low acousticness scores are linked to lower average popularity. This supports the hypothesis that a balance between synthetic and acoustic elements may be more appealing to listeners and is useful for optimizing track production features.

14. Top 20 Artists by Average Track Popularity (All Songs)
- Image: avg_popularity_per_artist.png
  
This horizontal bar chart shows the top 20 artists ranked by average track popularity, without filtering for a minimum number of songs. Compared to rationale #8 (which required ≥10 tracks per artist), this plot includes collaborations and artists with fewer songs, highlighting trending or viral artists. The inclusion of names like Sam Smith & Kim Petras, Beyoncé, and Rauw Alejandro underlines the impact of single releases and collabs on overall popularity. This visualization supports playlist curation, trend analysis, and reinforces the business case for spotlighting high-performing artist features regardless of catalog size.

15. Average Popularity by Liveness (Binned)
- Image: liveness_binned_avg_popularity.png
  
This bar chart presents the average track popularity across binned "liveness" scores, which estimate the presence of a live audience in a track. The popularity values remain relatively consistent across bins, with a subtle peak in the mid-liveness range (~0.1–0.13). This suggests that tracks perceived as moderately “live” may resonate slightly more with listeners, perhaps due to added energy or ambiance. However, extremes of low or high liveness don’t correlate strongly with popularity. This insight can inform decisions about production effects or live recording releases.

16. Average Popularity by Loudness (Binned)
- Image: loudness_binned_avg_popularity.png
  
This bar chart illustrates how average popularity changes across binned loudness values (in dB). The trend shows an increase in popularity from very quiet songs toward moderately loud ones (around 0.25–0.42 dB), after which popularity flattens and slightly decreases. This suggests that tracks with balanced loudness—not too soft or overly compressed—may perform better in terms of listener engagement. Useful for audio engineers and producers aiming for commercially viable mixes.

17. Average Popularity by Mode (Binned)
- Image: mode_binned_avg_popularity.png
  
This bar chart appears to show only a single aggregated bin, possibly due to preprocessing errors or imbalanced data between major (1) and minor (0) modes. With only one bin displayed, it's not possible to extract actionable differences in popularity between modes. It indicates the need for either data reprocessing or reassessment of this metric's variability across the dataset. Still, this highlights how categorical musical elements might need special handling in numeric visualizations.

18. Distribution of Songs by Mood & Energy
- Image: mood_energy_distribution.png
  
This horizontal bar chart categorizes songs into four combined mood/energy clusters: "Happy/High Energy," "Sad/Low Energy," "Energetic/Sad," and "Chill/Happy." The most prevalent segment is "Happy/High Energy," followed by “Sad/Low Energy.” This segmentation offers a high-level overview of the emotional and energetic diversity of the dataset. It is valuable for mood-based recommendation systems and playlist generators, guiding UX personalization strategies.

19. Average Popularity by Speechiness (Binned)
- Image: speechiness_binned_avg_popularity.png
  
This plot showcases the relationship between “speechiness” (a measure of spoken-word content) and average track popularity. The trend shows a mild downward curve, with highest popularity in low-speechiness ranges (~0.0–0.03), which aligns with standard musical tracks. Popularity dips as speechiness increases, reflecting how high speechiness (e.g., podcasts, spoken-word poetry) typically appeals to niche audiences. This has implications for genre classification and content targeting.

20. Average Popularity by Track Duration (Minutes, Binned)
- Image: duration_min_binned_avg_popularity.png
  
This bar chart bins track durations (in minutes) and examines their correlation with popularity. The most popular songs cluster around the 3.5 to 4-minute range, aligning with the typical radio-friendly length. Tracks that are too short or too long show a drop in average popularity. This reinforces existing industry norms and supports strategic editing for streaming optimization and listener retention.

21. Average Popularity by Track Duration (Milliseconds, Binned)
- Image: duration_ms_binned_avg_popularity.png
  
This visualization mirrors the previous one but offers greater precision in milliseconds. The same arc pattern appears, with popularity peaking in the mid-duration range (~210k–250k ms). This detail can be useful for algorithmic track trimming or segmentation strategies where exact timestamps matter—for instance, in audio previews, social media snippets, or music summarization tools.

22. Average Popularity by Energy (Binned)
- Image: energy_binned_avg_popularity.png
  
This chart highlights how average popularity varies across binned “energy” scores. The most popular tracks fall within the 0–0.5 energy range, peaking near moderate levels. High-energy tracks (above 1.0) tend to be less popular, suggesting listener preference for tracks that are dynamic yet not overly intense. Useful insight for production teams curating gym playlists, chill mixes, or studying genre-specific trends.

23. Average Popularity by Instrumentalness (Binned)
- Image: instrumentalness_binned_avg_popularity.png
  
This bar chart shows the inverse relationship between instrumentalness and popularity. Tracks with lower instrumentalness—meaning more vocals—tend to have higher average popularity, while fully instrumental songs see a drop. This suggests that vocals are a key driver of mass appeal in mainstream music, making this a crucial variable for both producers and recommendation systems.

24. Average Popularity by Key (Binned)
- Image: key_binned_avg_popularity.png
  
This chart examines the relationship between musical key and average popularity. While differences across key bins are subtle, there are slight peaks around certain keys (e.g., C and G). This could reflect the frequency of these keys in pop and mainstream production, possibly due to vocal comfort or instrument tuning. Although not a strong predictor of popularity, key distribution can inform harmonic mixing and automated DJ tools.
25. Average Popularity by Tempo (Binned)
- Image: tempo_binned_avg_popularity.png
  
This plot investigates how song tempo relates to popularity. Songs were binned by standardized tempo values, with the middle range (~0.82) achieving the highest average popularity. Extreme tempos, both fast and slow, show slight declines in popularity. This suggests that mid-tempo tracks may align better with mainstream listener preferences. This is a useful insight for producers aiming to optimize the tempo of new releases for broader appeal.

26. Average Popularity by Time Signature (Binned)
- Image: time_signature_binned_avg_popularity.png
  
This simple binned bar chart compares average popularity across different time signatures. Songs with uncommon time signatures (below 4/4) show slightly higher average popularity compared to standard 4/4 tracks. While the effect is subtle, it could suggest that rhythmically unique songs may stand out in the algorithmic and human listener environment. This insight can be explored further by niche or experimental artists.

27. Top 20 Artists by Average Popularity (All Songs)
- Image: top_artists_popularity.png
  
This horizontal bar chart highlights the top 20 artists by average popularity score, without filtering for a minimum number of tracks. This allows the inclusion of artists with just a few viral tracks or notable collaborations. The chart reflects how feature artists (e.g., Sam Smith & Kim Petras, Bad Bunny & Bomba Estéreo) can dominate popularity rankings through single releases rather than large catalogs. This supports the business case for highlighting top-performing track-level features in recommendation engines or collaboration decisions.


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

Tab 4: Track Predictor 
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

### Streamlit Web Application
Launch with: `streamlit run app.py`

#### Tab 1: Data Explorer 
- Browse 114,000 Spotify tracks
- Filter by genre and popularity range
- View track details (name, artists, audio features)
- Download filtered data as CSV

#### Tab 2: Visualizations 
- **Popularity Distribution**: Histogram of track popularity scores
- **Audio Features**: 6 feature distributions (danceability, energy, valence, etc.)
- **Top 20 Genres**: Bar chart of most popular genres
- **Mood & Energy**: Pie charts showing track classification
- **Correlation Heatmap**: Feature relationships
- **Interactive Scatter**: Custom X/Y axis selection with color coding

#### Tab 3: ML Model 
- **Model Performance Metrics**: R² = 0.39, RMSE = 17.4
- **Top 20 Feature Importances**: Genre, track length, explicit content
- **Prediction vs Actual**: Scatter plot visualization
- **Residual Analysis**: Error distribution
- **Complete Model Metadata**: JSON export of specifications

#### Tab 4: Track Predictor 
**For Music Producers**: Interactive tool to predict and optimize track popularity

**Input Methods**:
- Sliders (beginner-friendly with tooltips)
- Manual entry (advanced users)
- Random example (explore real tracks)

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
Thanks to our facilitator, Emma Lamont, and our  tutors for their help and support. 
