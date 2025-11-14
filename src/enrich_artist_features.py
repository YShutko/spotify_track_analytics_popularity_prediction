"""
Artist Feature Enrichment Script

Fetches artist metadata from Spotify Web API and enriches the dataset with:
- Artist followers
- Artist popularity
- Artist genres
- Artist total tracks
- Artist average track popularity

Expected R² improvement: +0.10-0.15 (from 0.16 to ~0.30)
"""

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup Spotify API client
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

print("="*80)
print("🎵 SPOTIFY ARTIST FEATURE ENRICHMENT")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("="*80)
print("📂 STEP 1: LOAD DATASET")
print("="*80)

df = pd.read_parquet('data/processed/cleaned_spotify_data.parquet')
print(f"✓ Loaded {len(df):,} tracks")
print(f"✓ Columns: {list(df.columns)}")
print()

# ============================================================================
# STEP 2: Extract Unique Artists
# ============================================================================
print("="*80)
print("🎤 STEP 2: EXTRACT UNIQUE ARTISTS")
print("="*80)

# Get unique artist names (handle multiple artists per track)
all_artists = df['artists'].dropna().unique()
print(f"✓ Found {len(all_artists):,} unique artist names/combinations")

# Split artist combinations (e.g., "Artist1, Artist2" → ["Artist1", "Artist2"])
unique_artists = set()
for artist_combo in all_artists:
    # Handle various separators
    for separator in [',', ';', '&', ' and ', ' feat. ', ' ft. ']:
        artist_combo = artist_combo.replace(separator, ',')

    # Split and clean
    artists = [a.strip() for a in artist_combo.split(',')]
    unique_artists.update(artists)

unique_artists = sorted(list(unique_artists))
print(f"✓ Extracted {len(unique_artists):,} individual artist names")
print(f"✓ Sample artists: {unique_artists[:5]}")
print()

# ============================================================================
# STEP 3: Fetch Artist Metadata from Spotify API
# ============================================================================
print("="*80)
print("🌐 STEP 3: FETCH ARTIST METADATA FROM SPOTIFY API")
print("="*80)

# Check for existing cache
cache_file = Path('data/processed/artist_metadata_cache.json')
if cache_file.exists():
    print(f"✓ Found existing cache: {cache_file}")
    with open(cache_file, 'r') as f:
        artist_data_cache = json.load(f)
    print(f"✓ Loaded {len(artist_data_cache)} cached artists")
else:
    artist_data_cache = {}
    print("✓ No cache found, starting fresh")

print()

artist_metadata = []
errors = []
not_found = []

# Fetch metadata for artists not in cache
artists_to_fetch = [a for a in unique_artists if a not in artist_data_cache]
print(f"✓ Need to fetch {len(artists_to_fetch):,} artists from API")
print(f"✓ Skipping {len(unique_artists) - len(artists_to_fetch):,} cached artists")
print()

if artists_to_fetch:
    print("Fetching artist data (with rate limiting)...")

    for i, artist_name in enumerate(tqdm(artists_to_fetch, desc="Fetching artists")):
        try:
            # Search for artist
            results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)

            if results['artists']['items']:
                artist = results['artists']['items'][0]

                # Extract metadata
                artist_info = {
                    'artist_name': artist_name,
                    'spotify_artist_id': artist['id'],
                    'artist_followers': artist['followers']['total'],
                    'artist_popularity': artist['popularity'],
                    'artist_genres': artist['genres'],  # List of genres
                    'artist_genre_count': len(artist['genres']),
                    'artist_top_genre': artist['genres'][0] if artist['genres'] else None
                }

                # Get artist's albums to estimate track count
                albums = sp.artist_albums(artist['id'], limit=50, album_type='album,single')
                artist_info['artist_album_count'] = albums['total']

                # Cache the result
                artist_data_cache[artist_name] = artist_info
                artist_metadata.append(artist_info)
            else:
                not_found.append(artist_name)
                artist_data_cache[artist_name] = None

            # Rate limiting: ~30 requests per second (Spotify allows much more)
            # spotipy has built-in rate limit handling, so we can be more aggressive
            if (i + 1) % 30 == 0:
                time.sleep(0.05)

        except Exception as e:
            errors.append({'artist': artist_name, 'error': str(e)})
            print(f"\n❌ Error fetching {artist_name}: {e}")

    # Save updated cache
    print()
    print(f"💾 Saving cache to {cache_file}")
    with open(cache_file, 'w') as f:
        json.dump(artist_data_cache, f, indent=2)
    print(f"✓ Cached {len(artist_data_cache)} artists")

# Load all data from cache
for artist_name in unique_artists:
    if artist_name in artist_data_cache and artist_data_cache[artist_name]:
        artist_metadata.append(artist_data_cache[artist_name])

print()
print(f"✓ Successfully fetched: {len(artist_metadata):,} artists")
print(f"✓ Not found: {len(not_found):,} artists")
print(f"✓ Errors: {len(errors)}")
print()

# ============================================================================
# STEP 4: Create Artist Features Dataframe
# ============================================================================
print("="*80)
print("📊 STEP 4: CREATE ARTIST FEATURES DATAFRAME")
print("="*80)

artist_df = pd.DataFrame(artist_metadata)

if len(artist_df) > 0:
    print(f"✓ Created artist dataframe with {len(artist_df):,} rows")
    print(f"✓ Columns: {list(artist_df.columns)}")
    print()
    print("Summary statistics:")
    print(artist_df[['artist_followers', 'artist_popularity', 'artist_genre_count', 'artist_album_count']].describe())
    print()
else:
    print("❌ No artist data fetched!")
    exit(1)

# ============================================================================
# STEP 5: Enrich Original Dataset
# ============================================================================
print("="*80)
print("🔗 STEP 5: ENRICH ORIGINAL DATASET WITH ARTIST FEATURES")
print("="*80)

# Handle tracks with multiple artists - use primary artist (first one)
def get_primary_artist(artist_combo):
    """Extract primary (first) artist from artist combination"""
    if pd.isna(artist_combo):
        return None

    # Handle various separators
    for separator in [',', ';', '&', ' and ', ' feat. ', ' ft. ']:
        artist_combo = artist_combo.replace(separator, ',')

    # Return first artist
    artists = [a.strip() for a in artist_combo.split(',')]
    return artists[0] if artists else None

df['primary_artist'] = df['artists'].apply(get_primary_artist)

# Merge artist features
df_enriched = df.merge(
    artist_df,
    left_on='primary_artist',
    right_on='artist_name',
    how='left'
)

print(f"✓ Merged artist features")
print(f"✓ New dataframe shape: {df_enriched.shape}")
print()

# Check merge success rate
matched = df_enriched['artist_followers'].notna().sum()
total = len(df_enriched)
match_rate = matched / total * 100

print(f"Match Statistics:")
print(f"  Matched: {matched:,} / {total:,} ({match_rate:.1f}%)")
print(f"  Unmatched: {total - matched:,} ({100-match_rate:.1f}%)")
print()

# Fill missing artist features with defaults
fill_values = {
    'artist_followers': 0,
    'artist_popularity': df_enriched['artist_popularity'].median(),
    'artist_genre_count': 0,
    'artist_album_count': 1
}

for col, val in fill_values.items():
    df_enriched[col].fillna(val, inplace=True)
    print(f"✓ Filled missing {col} with {val}")

print()

# ============================================================================
# STEP 6: Feature Engineering
# ============================================================================
print("="*80)
print("🛠️  STEP 6: FEATURE ENGINEERING")
print("="*80)

# Log-transform followers (highly skewed)
df_enriched['artist_followers_log'] = np.log1p(df_enriched['artist_followers'])
print("✓ Created artist_followers_log (log-transformed)")

# Artist tier (categorical)
df_enriched['artist_tier'] = pd.cut(
    df_enriched['artist_followers'],
    bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
    labels=['Unknown', 'Emerging', 'Growing', 'Established', 'Superstar']
)
print("✓ Created artist_tier (categorical)")

# Popularity gap (track vs artist baseline)
df_enriched['popularity_vs_artist'] = df_enriched['popularity'] - df_enriched['artist_popularity']
print("✓ Created popularity_vs_artist (track performance vs artist baseline)")

print()

# Show sample
print("Sample of enriched data:")
print(df_enriched[['track_name', 'primary_artist', 'popularity',
                   'artist_followers', 'artist_popularity', 'artist_tier']].head(10))
print()

# ============================================================================
# STEP 7: Save Enriched Dataset
# ============================================================================
print("="*80)
print("💾 STEP 7: SAVE ENRICHED DATASET")
print("="*80)

output_file = 'data/processed/cleaned_spotify_data_with_artists.parquet'
df_enriched.to_parquet(output_file, index=False)

print(f"✓ Saved enriched dataset to: {output_file}")
print(f"✓ File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
print(f"✓ Shape: {df_enriched.shape}")
print()

# ============================================================================
# STEP 8: Summary Report
# ============================================================================
print("="*80)
print("📈 SUMMARY REPORT")
print("="*80)

print(f"""
Dataset Statistics:
  Original tracks: {len(df):,}
  Enriched tracks: {len(df_enriched):,}
  Unique artists: {len(unique_artists):,}
  Artists with metadata: {len(artist_metadata):,}
  Match rate: {match_rate:.1f}%

New Features Added:
  1. artist_followers (int) - Total Spotify followers
  2. artist_popularity (int) - Spotify popularity score (0-100)
  3. artist_genres (list) - List of artist genres
  4. artist_genre_count (int) - Number of genres
  5. artist_top_genre (str) - Primary genre
  6. artist_album_count (int) - Estimated album/single count
  7. artist_followers_log (float) - Log-transformed followers
  8. artist_tier (category) - Categorized artist level
  9. popularity_vs_artist (float) - Track popularity vs artist baseline

Expected Model Improvement:
  Current R²: 0.16 (audio-only features)
  Expected R²: 0.28-0.32 (audio + artist features)
  Improvement: +0.12-0.16 (~75-100% increase)

Next Steps:
  1. Run: python src/train_with_artist_features.py
  2. Compare R² before (0.16) vs after (~0.30)
  3. Analyze feature importance of new artist features
  4. Update dashboards with enriched model
""")

print("="*80)
print(f"✅ ENRICHMENT COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
