![Music Discovery Banner](./image/banner.png)

# Music Discovery & Song Recommendation System

A comprehensive machine learning project that combines music data analysis, clustering algorithms, and Spotify API integration to build an intelligent song recommendation system. This project uses unsupervised learning techniques to discover music patterns and provide personalized song recommendations based on audio features.

## Project Overview

This project implements a sophisticated music recommendation system that:

- **Analyzes Music Data**: Processes large datasets of songs with audio features (acousticness, danceability, energy, etc.)
- **Clusters Similar Music**: Uses K-Means clustering to group songs and genres by their audio characteristics
- **Visualizes Music Patterns**: Creates interactive visualizations using dimensionality reduction techniques (t-SNE, PCA)
- **Provides Recommendations**: Implements a content-based recommendation engine using cosine similarity
- **Integrates Spotify API**: Retrieves real-time music data and audio features from Spotify's extensive catalog

The system analyzes musical features such as:
- **Valence**: Musical positiveness (happy vs sad)
- **Energy**: Intensity and power of the track
- **Danceability**: How suitable a track is for dancing
- **Acousticness**: Whether the track is acoustic
- **Instrumentalness**: Predicts whether a track contains vocals
- **Liveness**: Detects the presence of an audience
- **Speechiness**: Detects spoken words in a track
- **Tempo**: Speed/pace of the music

## Technical Stack

### Core Libraries & Frameworks
- **Python 3.7+**: Primary programming language
- **Jupyter Notebook**: Interactive development environment
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis

### Machine Learning & Data Science
- **Scikit-learn**: 
  - K-Means clustering algorithms
  - StandardScaler for feature normalization
  - Pipeline for streamlined ML workflows
  - t-SNE for dimensionality reduction
  - PCA for principal component analysis
  - Distance metrics (Euclidean, Cosine similarity)
- **SciPy**: Advanced scientific computing and distance calculations

### Data Visualization
- **Plotly Express**: Interactive web-based visualizations
- **Matplotlib**: Static plotting and visualization
- **Seaborn**: Statistical data visualization

### External APIs & Services
- **Spotipy**: Official Spotify Web API Python wrapper
- **Spotify Web API**: Access to Spotify's music catalog and audio features
- **Google Colab Drive**: Cloud storage integration for datasets

### Development Environment
- **Google Colaboratory**: Cloud-based Jupyter notebook environment
- **VS Code**: Local development and notebook editing

## Data Sources

### Primary Datasets
The project uses the **Spotify Dataset** from Kaggle: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset

This dataset provides three main CSV files:

1. **`data.csv`**: Main music dataset containing individual song records
   - Song metadata (name, artist, year, popularity)
   - Audio features (valence, energy, danceability, etc.)
   - Technical attributes (duration, key, mode, tempo)

2. **`data_by_genres.csv`**: Genre-aggregated music data
   - Average audio features per genre
   - Genre classification and clustering data

3. **`data_by_year.csv`**: Time-series music data
   - Yearly trends in music features
   - Evolution of musical characteristics over time

### Data Schema
The system expects the following key features for each song:
```
- valence: Musical positiveness (0.0 to 1.0)
- year: Release year
- acousticness: Acoustic confidence measure (0.0 to 1.0)
- danceability: Dance suitability (0.0 to 1.0)
- duration_ms: Track length in milliseconds
- energy: Intensity measure (0.0 to 1.0)
- explicit: Explicit content flag (0 or 1)
- instrumentalness: Vocal absence prediction (0.0 to 1.0)
- key: Musical key (0-11)
- liveness: Live audience detection (0.0 to 1.0)
- loudness: Overall loudness in decibels
- mode: Musical modality (0 = minor, 1 = major)
- popularity: Spotify popularity score (0-100)
- speechiness: Spoken word detection (0.0 to 1.0)
- tempo: Beats per minute
```

### Spotify API Integration
- **Client Credentials Flow**: Secure API authentication
- **Track Search**: Real-time song lookup by name and year
- **Audio Features**: Retrieval of detailed audio analysis
- **Track Metadata**: Access to popularity, duration, and other metrics

## Machine Learning Workflow

### 1. Data Preprocessing
- **Feature Selection**: Automatic selection of numerical features
- **Standardization**: Z-score normalization using StandardScaler
- **Data Cleaning**: Handling missing values and outliers

### 2. Clustering Analysis

#### Genre Clustering
- **Algorithm**: K-Means with 10 clusters
- **Purpose**: Group similar music genres
- **Features**: All numerical audio features
- **Visualization**: t-SNE projection for 2D visualization

#### Song Clustering
- **Algorithm**: K-Means with 20 clusters
- **Purpose**: Group individual songs by similarity
- **Features**: 15 audio features
- **Scalability**: Pipeline approach for efficient processing

### 3. Dimensionality Reduction

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Components**: 2D projection
- **Purpose**: Visualize genre clustering patterns
- **Parameters**: `random_state=42` for reproducibility

#### PCA (Principal Component Analysis)
- **Components**: 2D projection
- **Purpose**: Visualize song clustering in reduced space
- **Application**: Song embedding visualization

### 4. Recommendation System

#### Content-Based Filtering
- **Similarity Metric**: Cosine similarity
- **Feature Space**: 15-dimensional audio feature vectors
- **Input**: Single song or playlist of songs
- **Output**: Top-N similar songs

#### Recommendation Process
1. **Input Processing**: Accept song name and year
2. **Feature Extraction**: Get audio features from dataset or Spotify API
3. **Vector Averaging**: Calculate mean vector for playlist input
4. **Similarity Calculation**: Compute cosine distances to all songs
5. **Ranking**: Sort by similarity and return top recommendations
6. **Filtering**: Remove input songs from recommendations

## Getting Started

### Prerequisites
```bash
# Python 3.7 or higher
python --version

# Required for local development
pip install jupyter notebook
```

### Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd music-discovery
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy pandas scikit-learn plotly matplotlib seaborn spotipy scipy
   ```

3. **Spotify API Setup**
   - Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com)
   - Create a new app to get Client ID and Client Secret
   - Replace the credentials in the notebook:
   ```python
   client_id = 'your_client_id_here'
   client_secret = 'your_client_secret_here'
   ```

4. **Data Setup**
   - Place your CSV files in the appropriate directory
   - Update file paths in the notebook:
   ```python
   data_path = 'path/to/your/data.csv'
   genre_data_path = 'path/to/your/data_by_genres.csv'
   year_data_path = 'path/to/your/data_by_year.csv'
   ```

### Running the Project

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the Notebook**
   - Navigate to `Retrieval Song Recommendation.ipynb`
   - Run cells sequentially from top to bottom

3. **Execute Analysis**
   - Load and explore the datasets
   - Perform clustering analysis
   - Generate visualizations
   - Test the recommendation system

## Usage Examples

### Basic Song Recommendation
```python
# Get recommendations for a single song
recommendations = recommend_songs([{'name': 'Basket Case', 'year': 1994}], data)
print(recommendations)
```

### Playlist-Based Recommendations
```python
# Get recommendations based on multiple songs
playlist = [
    {'name': 'Bohemian Rhapsody', 'year': 1975},
    {'name': 'Stairway to Heaven', 'year': 1971},
    {'name': 'Hotel California', 'year': 1977}
]
recommendations = recommend_songs(playlist, data, n_songs=15)
```

### Finding Song Information
```python
# Search for a song using Spotify API
song_info = find_song('Yesterday', 1965)
print(song_info)
```

## Visualizations

The project generates several interactive visualizations:

1. **Genre Clustering Visualization**
   - 2D t-SNE projection of music genres
   - Color-coded clusters
   - Interactive hover information

2. **Song Clustering Visualization**
   - 2D PCA projection of individual songs
   - Cluster-based coloring
   - Song title hover data

3. **Feature Distribution Analysis**
   - Statistical plots of audio features
   - Genre comparison charts
   - Temporal trend analysis

## Technical Implementation Details

### Clustering Pipeline
```python
# Genre clustering pipeline
cluster_pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=10))

# Song clustering pipeline
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])
```

### Recommendation Algorithm
```python
def recommend_songs(song_list, spotify_data, n_songs=10):
    # Calculate mean vector of input songs
    song_center = get_mean_vector(song_list, spotify_data)
    
    # Scale features and calculate cosine distances
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    
    # Return top N recommendations
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    return rec_songs[metadata_cols].to_dict(orient='records')
```

## Configuration

### Clustering Parameters
- **Genre Clusters**: 10 (adjustable based on dataset size)
- **Song Clusters**: 20 (scalable for larger datasets)
- **Random State**: 42 (for reproducible results)

### Recommendation Parameters
- **Default Recommendations**: 10 songs
- **Distance Metric**: Cosine similarity
- **Feature Normalization**: Z-score standardization

### Visualization Settings
- **t-SNE Perplexity**: Default sklearn settings
- **PCA Components**: 2 for visualization
- **Plot Interactivity**: Enabled via Plotly

## Resources

- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Music Information Retrieval Research](https://musicinformationretrieval.com/)
- [Audio Feature Analysis Papers](https://scholar.google.com/scholar?q=audio+feature+analysis+music)

---

*This project demonstrates the intersection of machine learning, music analysis, and recommendation systems, providing insights into both technical implementation and musical pattern discovery.*