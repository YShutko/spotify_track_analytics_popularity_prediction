# Spotify Track Analytics Popularity Prediction

# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

<p align="center">
  <img src="ChatGPT Image Nov 11, 2025, 11_08_22 AM.png" alt="Project Logo" width="25%">
</p>


## Content
* [Readme.md](https://github.com/YShutko/spotify_track_analytics_popularity_prediction/blob/1eb084f166f61e2ec0c6dcf23cdb3fea6f7f3cb8/README.md)
* [Kanban Project Board](https://github.com/users/YShutko/projects/6)
* [Datasets](https://github.com/YShutko/spotify_track_analytics_popularity_prediction/tree/926c78849c653732de5f592223b0ff7cda01fab8/data) 
* [Jupyter notebook]()

## Dataset Content
The data set used for this project: [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). TA collection of ~114,000 songs across 125 genres with features like danceability, energy, tempo, and popularity. Ideal for audio analysis, genre classification, and music trend exploration.

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

## Business Requirements
1. **Understand Key Drivers of Song Popularity**  
   Analyze which audio features (e.g., danceability, energy, tempo, valence) have the strongest influence on a song’s popularity score.
2. **Classify Songs by Mood and Energy**  
   Segment songs into categories such as *Happy*, *Sad*, *Chill*, or *High Energy* using features like valence, energy, and tempo.
3. **Genre-Level Analysis**  
   Identify trends across different music genres — for example, which genres are more danceable, louder, or more acoustic.
4. **Support Playlist Curation**  
   Enable smarter playlist generation by grouping tracks based on shared characteristics, allowing users to build playlists for specific moods or activities.
5. **Data-Driven Music Recommendations**  
   Establish the foundation for future ML-powered recommendations by exploring the relationship between track features and user listening behavior.


## Hypothesis


## Project Plan
* Data inspection & cleaning: preparing categorical and numerical features for analysis.
* Univariate analysis: target variable (Attrition_Flag) distribution.
* Bivariate analysis: exploring relationships between features and churn (Attrition_Flag).
* Correlation analysis: comparing numerical relationships across existing and attrited customers.
* Group comparisons: using difference heatmaps to highlight how patterns shift with churn.
* Visualization: interactive and static plots to present insights clearly.

## The rationale to map the business requirements to the Data Visualisations


## Dashboard Design
  
## Analysis techniques used
* Visual Studio Code
* Python
* Jupyter notebook
* ChatGPT

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

## Main Data Analysis Libraries
* Pandas
* Numpy
* Plotly
* Seabon
* Matplotlib
* Gradio
 
## Credits 
* [The Code Institute](https://codeinstitute.net/) Learning Management System
* [VS Code](https://code.visualstudio.com/) was used to wite the code
* [ChatGPT](https://chatgpt.com/) was used to generate and debug code
* [README](https://github.com/Code-Institute-Solutions/da-README-template) template
* [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) data set was used for this project
* [gradio](https://www.gradio.app/)

# Acknowledgements
Thanks to our facilitator Emma Lamont and  tutors for their help and support. 
