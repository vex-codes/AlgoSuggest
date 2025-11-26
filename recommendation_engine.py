import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import pickle
import os
import json
import difflib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---
DATASET_PATH = './tmdb_5000_movies.csv'
COSINE_SIM_PATH = 'cosine_sim.npz'  # Changed to .npz for sparse matrix
INDICES_PATH = 'indices.pkl'
METADATA_PATH = 'metadata_processed.pkl'
SPARSITY_THRESHOLD = 0.01  # Threshold for zeroing out small similarity values

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to hold model data
cosine_sim = None
indices = None
df = None

def load_data():
    """
    Loads the dataset from a CSV file.
    Validates that the file contains necessary columns.
    """
    df = pd.DataFrame()
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            # Check for critical columns
            required_columns = ['overview', 'title']
            if not all(col in df.columns for col in required_columns):
                print(f"WARNING: '{DATASET_PATH}' exists but is missing required columns: {required_columns}.")
                print(f"   Found columns: {list(df.columns)}")
                print("   It looks like you might have the 'credits' file instead of the 'movies' file.")
                print("   Falling back to dummy data.")
                df = pd.DataFrame() # Trigger fallback
        except Exception as e:
            print(f"Error loading dataset: {e}")
            df = pd.DataFrame()

    if df.empty:
        # Create a dummy dataset for demonstration if the file doesn't exist or is invalid
        print("Creating a small dummy dataset for demonstration purposes.")
        data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction', 'The Godfather', 'Toy Story', 'Finding Nemo', 'The Avengers', 'Iron Man'],
            'genres': ['[{"name": "Action"}, {"name": "Sci-Fi"}]', '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Sci-Fi"}]', '[{"name": "Adventure"}, {"name": "Drama"}, {"name": "Sci-Fi"}]', '[{"name": "Action"}, {"name": "Crime"}, {"name": "Drama"}]', '[{"name": "Crime"}, {"name": "Drama"}]', '[{"name": "Crime"}, {"name": "Drama"}]', '[{"name": "Animation"}, {"name": "Adventure"}, {"name": "Comedy"}]', '[{"name": "Animation"}, {"name": "Adventure"}, {"name": "Comedy"}]', '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Sci-Fi"}]', '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Sci-Fi"}]'],
            'overview': [
                'A computer hacker learns from mysterious rebels about the true nature of his reality.',
                'A thief who steals corporate secrets through the use of dream-sharing technology.',
                'A team of explorers travel through a wormhole in space.',
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham.',
                'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine.',
                'The aging patriarch of an organized crime dynasty transfers control.',
                'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him.',
                'After his son is captured in the Great Barrier Reef, a timid clownfish sets out on a journey.',
                'Earth\'s mightiest heroes must come together and learn to fight as a team.',
                'After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a suit of armor.'
            ],
            # Add dummy voting data for Weighted Rating
            'vote_count': [1000, 900, 800, 950, 850, 900, 700, 750, 800, 850],
            'vote_average': [8.7, 8.8, 8.6, 9.0, 8.9, 9.2, 8.3, 8.1, 8.0, 7.9],
            'keywords': ['[{"name": "hacker"}]', '[{"name": "dream"}]', '[{"name": "space"}]', '[{"name": "dc comics"}]', '[{"name": "gangster"}]', '[{"name": "mafia"}]', '[{"name": "toy"}]', '[{"name": "fish"}]', '[{"name": "marvel"}]', '[{"name": "superhero"}]']
        }
        return pd.DataFrame(data)
    
    return df

def parse_list_column(json_str):
    """
    Parses a JSON string representation of a list of dicts and extracts the 'name' field.
    """
    if not isinstance(json_str, str):
        return ""
    try:
        data = json.loads(json_str)
        names = [item['name'] for item in data if 'name' in item]
        return " ".join(names)
    except:
        return ""

def clean_data(text):
    """
    Cleans the input text by converting to lowercase and stripping whitespace.
    """
    if isinstance(text, str):
        return text.lower().strip()
    return ""

def calculate_weighted_rating(df, m=None, C=None):
    """
    Calculates the IMDB Weighted Rating for each movie.
    """
    if C is None:
        C = df['vote_average'].mean()
    
    if m is None:
        m = df['vote_count'].quantile(0.9)
        
    v = df['vote_count']
    R = df['vote_average']
    
    df['weighted_rating'] = (v / (v + m) * R) + (m / (v + m) * C)
    return df

def preprocess_data(df):
    """
    Preprocesses the DataFrame.
    """
    print("   - Filling nulls...")
    df['overview'] = df['overview'].fillna('')
    
    print("   - Extracting genres and keywords...")
    if 'genres' in df.columns:
        df['clean_genres'] = df['genres'].apply(parse_list_column)
        df['clean_genres'] = df['clean_genres'].apply(clean_data)
    else:
        df['clean_genres'] = ""

    if 'keywords' in df.columns:
        df['clean_keywords'] = df['keywords'].apply(parse_list_column)
        df['clean_keywords'] = df['clean_keywords'].apply(clean_data)
    else:
        df['clean_keywords'] = ""

    df['clean_overview'] = df['overview'].apply(clean_data)
    
    print("   - Creating feature soup...")
    df['features_soup'] = df['clean_genres'] + ' ' + df['clean_keywords'] + ' ' + df['clean_overview']
    
    if 'vote_count' in df.columns and 'vote_average' in df.columns:
        print("   - Calculating weighted ratings...")
        df = calculate_weighted_rating(df)
    else:
        df['weighted_rating'] = 0
        
    return df

def get_recommendations(title, cosine_sim, df, indices, N=10):
    """
    Get top N recommendations for a given title.
    Uses fuzzy matching if exact title is not found.
    Works with both dense and sparse cosine similarity matrices.
    """
    # Handle case sensitivity
    title_map = {str(t).lower(): t for t in df['title']}
    
    search_title = title.lower()
    
    # Fuzzy matching
    if search_title not in title_map:
        # Get list of all titles
        all_titles = list(title_map.keys())
        # Find closest matches
        matches = difflib.get_close_matches(search_title, all_titles, n=1, cutoff=0.4)
        if matches:
            search_title = matches[0]
            print(f"   Fuzzy match: '{title}' -> '{title_map[search_title]}'")
        else:
            return [] # No match found
    
    exact_title = title_map[search_title]
    
    if exact_title not in indices:
         return []

    idx = indices[exact_title]
    
    # Handle sparse matrix - convert row to array
    if sparse.issparse(cosine_sim):
        sim_scores = list(enumerate(cosine_sim[idx].toarray()[0]))
    else:
        sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices].tolist()

def get_genre_recommendations(target_genre, df, N=10):
    """
    Get top N recommendations based on a target genre, sorted by Weighted Rating.
    """
    # Fix for 'Sci-Fi' vs 'Science Fiction'
    genre_mapping = {
        'sci-fi': 'science fiction',
        'sci fi': 'science fiction',
        'sf': 'science fiction'
    }
    
    target_genre = target_genre.lower()
    target_genre = genre_mapping.get(target_genre, target_genre)
    
    mask = df['clean_genres'].str.contains(target_genre)
    genre_df = df[mask]
    
    if genre_df.empty:
        return []
    
    genre_df = genre_df.sort_values('weighted_rating', ascending=False)
    
    return genre_df['title'].head(N).tolist()

# --- API Routes ---

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    title = data.get('title', '')
    if not title:
        return jsonify({'error': 'Title is required'}), 400
    
    recs = get_recommendations(title, cosine_sim, df, indices)
    
    if not recs:
        return jsonify({'error': f"Sorry, we couldn't find any recommendations for '{title}'. Try checking the spelling!"}), 404
        
    return jsonify({'recommendations': recs})

@app.route('/api/recommend_genre', methods=['POST'])
def api_recommend_genre():
    data = request.json
    genre = data.get('genre', '')
    if not genre:
        return jsonify({'error': 'Genre is required'}), 400
    
    recs = get_genre_recommendations(genre, df)
    
    if not recs:
         return jsonify({'error': f"No movies found for genre '{genre}'."}), 404

    return jsonify({'recommendations': recs})

@app.route('/api/autocomplete', methods=['GET'])
def api_autocomplete():
    query = request.args.get('query', '').lower()
    if not query or len(query) < 2:
        return jsonify({'suggestions': []})
    
    # Simple substring match for autocomplete
    # Limit to top 5 for performance
    matches = df[df['title'].str.lower().str.contains(query, na=False)]['title'].head(5).tolist()
    return jsonify({'suggestions': matches})

def initialize_model():
    global df, cosine_sim, indices
    
    print("1. Loading data...")
    reprocess = True
    if os.path.exists(METADATA_PATH) and os.path.exists(DATASET_PATH):
        if os.path.getmtime(DATASET_PATH) < os.path.getmtime(METADATA_PATH):
            reprocess = False
            
    if not reprocess:
        print("   Loading processed metadata from disk (cache hit)...")
        df = pd.read_pickle(METADATA_PATH)
    else:
        print("   Processing raw data (cache miss or update detected)...")
        df = load_data()
        if df.empty:
            print("   Exiting due to empty dataset.")
            exit()
        print(f"   Loaded {len(df)} items.")
        print("2. Preprocessing data...")
        df = preprocess_data(df)
        print("   Saving processed metadata...")
        df.to_pickle(METADATA_PATH)

    recompute_model = True
    if os.path.exists(COSINE_SIM_PATH) and os.path.exists(INDICES_PATH):
        if os.path.exists(METADATA_PATH):
             if os.path.getmtime(METADATA_PATH) < os.path.getmtime(COSINE_SIM_PATH):
                 recompute_model = False
        else:
             recompute_model = False

    if not recompute_model:
        print("3. Loading pre-computed model...")
        cosine_sim = sparse.load_npz(COSINE_SIM_PATH)
        with open(INDICES_PATH, 'rb') as f:
            indices = pickle.load(f)
        print("   Model loaded successfully.")
    else:
        print("3. Computing model (this may take a moment)...")
        tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        tfidf_matrix = tfidf.fit_transform(df['features_soup'])
        print(f"   TF-IDF Matrix shape: {tfidf_matrix.shape}")
        
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Convert to sparse matrix to save space
        print(f"   Converting to sparse matrix (threshold={SPARSITY_THRESHOLD})...")
        # Zero out small values to increase sparsity
        cosine_sim[cosine_sim < SPARSITY_THRESHOLD] = 0
        cosine_sim_sparse = sparse.csr_matrix(cosine_sim)
        
        # Calculate sparsity
        sparsity = 1.0 - (cosine_sim_sparse.nnz / (cosine_sim_sparse.shape[0] * cosine_sim_sparse.shape[1]))
        print(f"   Sparsity: {sparsity*100:.2f}% (storing only {(1-sparsity)*100:.2f}% of values)")
        
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        print("   Saving model to disk...")
        sparse.save_npz(COSINE_SIM_PATH, cosine_sim_sparse)
        with open(INDICES_PATH, 'wb') as f:
            pickle.dump(indices, f)
        print("   Model saved.")
        
        # Update the global variable to use sparse matrix
        cosine_sim = cosine_sim_sparse

# Initialize model at module level for Gunicorn workers
print("--- AlgoSuggest Engine (Initializing) ---")
initialize_model()

if __name__ == "__main__":
    print("--- AlgoSuggest Engine (Dev Server) ---")
    # initialize_model() # Already called above
    print("\nStarting Flask API Server on port 5000...")
    print("You can now open 'index.html' in your browser.")
    app.run(debug=True, port=5000)
