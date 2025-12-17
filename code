import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION & API ---
# Fixed the SyntaxError by adding quotes around the key
TMDB_API_KEY = "6aa80f34b5ad28468201d1ad776e221e" 

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_showtimes_url(movie_title):
    query = f"{movie_title} movie showtimes near me"
    query_formatted = query.replace(" ", "+")
    return f"https://www.google.com/search?q={query_formatted}"

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    # Make sure 'movies_metadata.csv' is uploaded to the same folder on GitHub!
    df = pd.read_csv('movies_metadata.csv') 
    df['combined_features'] = df['genres'].fillna('') + " " + df['overview'].fillna('')
    return df

# --- 3. RECOMMENDATION ENGINE ---
def recommend(user_input, df):
    tfidf = TfidfVectorizer(stop_words='english')
    temp_df = pd.concat([df, pd.DataFrame({'combined_features': [user_input]})], ignore_index=True)
    tfidf_matrix = tfidf.fit_transform(temp_df['combined_features'])
    sim_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    movie_indices = sim_scores.argsort()[0][-5:][::-1]
    return df.iloc[movie_indices]

# --- 4. USER INTERFACE ---
st.set_page_config(page_title="Movie Matchmaker", layout="wide")
st.title("🎬 Multi-Domain Movie Discovery")

with st.sidebar:
    st.header("What do you like?")
    books = st.text_input("Favorite Books")
    music = st.text_input("Favorite Music")
    movies = st.text_input("Favorite Movies")
    submit = st.button("Find My Next Movie")

if submit:
    df = load_data()
    user_profile = f"{books} {music} {movies}"
    recommendations = recommend(user_profile, df)
    
    st.subheader("Your Personalized Recommendations")
    cols = st.columns(5)
    
    for i, (index, row) in enumerate(recommendations.iterrows()):
        with cols[i]:
            # Requirement: Fetch live posters
            poster = fetch_poster(row['id'])
            st.image(poster)
            st.markdown(f"**{row['title']}**")
            
            # Requirement: Find local showtimes
            url = get_showtimes_url(row['title'])
            st.markdown(f"[📍 View Showtimes]({url})")
