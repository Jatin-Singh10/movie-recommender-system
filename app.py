import pickle
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Replace with your real OMDb API key only
OMDB_API_KEY = "da63664"

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f1a;
        color: white;
    }

    .main-title {
        text-align: center;
        color: red;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .sub-title {
        text-align: center;
        color: #d1d5db;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .movie-card-title {
        text-align: center;
        font-weight: bold;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background-color: transparent !important;
        color: #ff4b6e !important;
        border: 2px solid #ff4b6e !important;
        border-radius: 10px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }

    .stButton > button:hover {
        background-color: rgba(255, 75, 110, 0.1) !important;
        color: #ff6b88 !important;
        border-color: #ff6b88 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    with open("artifacts/df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("artifacts/indices.pkl", "rb") as f:
        indices = pickle.load(f)

    with open("artifacts/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return df, indices, tfidf_matrix

df, indices, tfidf_matrix = load_data()

# -------------------- OMDb POSTER FUNCTION --------------------
def fetch_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()

        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return data.get("Poster")
    except Exception:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"

# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie_title):
    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:9]

    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices].tolist()

# -------------------- UI --------------------
st.markdown("<div class='main-title'>🎬 Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Search your favorite movie and get top recommendations</div>", unsafe_allow_html=True)

movie_list = df["title"].dropna().tolist()
selected_movie = st.selectbox("Search your favorite movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("🔥 Top Recommendations")

    cols = st.columns(4)

    for i, movie in enumerate(recommendations):
        poster = fetch_poster(movie)

        with cols[i % 4]:
            st.image(poster, use_container_width=True)
            st.markdown(
                f"<div class='movie-card-title'>{movie}</div>",
                unsafe_allow_html=True
            )