import difflib
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "artifacts"


@st.cache_resource
def load_artifacts():
    df_path = DATA_DIR / "df.pkl"
    tfidf_matrix_path = DATA_DIR / "tfidf_matrix.pkl"
    indices_path = DATA_DIR / "indices.pkl"

    missing = [
        str(p.name)
        for p in [df_path, tfidf_matrix_path, indices_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing artifact files in /artifacts: " + ", ".join(missing)
        )

    df = pd.read_pickle(df_path)
    with open(tfidf_matrix_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(indices_path, "rb") as f:
        indices = pickle.load(f)

    df = df.reset_index(drop=True)
    title_lookup = {str(title).strip().lower(): i for i, title in enumerate(df["title"].fillna(""))}
    titles = df["title"].fillna("").astype(str).tolist()
    return df, tfidf_matrix, indices, title_lookup, titles


def find_best_match(query: str, titles: list[str], title_lookup: dict[str, int]):
    normalized = query.strip().lower()
    if normalized in title_lookup:
        return titles[title_lookup[normalized]], []

    close = difflib.get_close_matches(query, titles, n=5, cutoff=0.5)
    if close:
        return close[0], close
    return None, []


@st.cache_data(show_spinner=False)
def recommend_movies(selected_title: str, top_n: int = 8):
    df, tfidf_matrix, _, title_lookup, titles = load_artifacts()
    matched_title, suggestions = find_best_match(selected_title, titles, title_lookup)

    if not matched_title:
        return None, suggestions, None

    idx = title_lookup[matched_title.strip().lower()]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1 : top_n + 1]

    result = df.iloc[similar_idx].copy()
    result["similarity_score"] = sim_scores[similar_idx]
    cols = [
        c
        for c in ["title", "genres", "tagline", "overview", "vote_average", "popularity", "similarity_score"]
        if c in result.columns
    ]
    return matched_title, suggestions, result[cols], idx


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
    }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(236,72,153,0.18));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        color: white;
        font-size: 2.4rem;
    }
    .hero p {
        color: #e5e7eb;
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    .metric-box {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        color: white;
    }
    .movie-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
    }
    .movie-card h3 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        color: #f9fafb;
    }
    .meta {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-bottom: 0.5rem;
    }
    .tagline {
        color: #fbcfe8;
        font-style: italic;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎬 Movie Recommendation System</h1>
        <p>Search a movie title and get similar recommendations using TF-IDF + cosine similarity on overview, genres, and taglines.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df, tfidf_matrix, indices, title_lookup, titles = load_artifacts()
except Exception as e:
    st.error(f"Unable to load deployment artifacts: {e}")
    st.info(
        "Place df.pkl, tfidf_matrix.pkl, and indices.pkl inside a folder named artifacts next to app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Search Settings")
    movie_query = st.text_input("Enter movie name", placeholder="e.g. Avatar, The Dark Knight, Interstellar")
    top_n = st.slider("Number of recommendations", min_value=5, max_value=15, value=8)
    run_search = st.button("Recommend Movies", use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Snapshot")
    st.write(f"Total movies: **{len(df):,}**")
    if "genres" in df.columns:
        st.write(f"Non-empty genre rows: **{df['genres'].astype(str).ne('').sum():,}**")
    st.write("Tip: even if the exact title is not found, the app will try to suggest close matches.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-box"><h3>{len(df):,}</h3><p>Movies Indexed</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><h3>{tfidf_matrix.shape[1]:,}</h3><p>TF-IDF Features</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><h3>{min(50000, tfidf_matrix.shape[1]):,}</h3><p>Max Features Cap</p></div>', unsafe_allow_html=True)

if not movie_query:
    st.subheader("Try these sample titles")
    sample_titles = df["title"].dropna().astype(str).head(12).tolist()
    st.write(" | ".join(sample_titles))

if run_search and movie_query:
    matched_title, suggestions, recommendations, selected_idx = recommend_movies(movie_query, top_n)

    if recommendations is None:
        st.warning("Movie not found in the index.")
        if suggestions:
            st.info("Closest matches: " + ", ".join(suggestions))
    else:
        st.success(f"Showing recommendations for: {matched_title}")
        source_movie = df.iloc[selected_idx]

        st.subheader("Selected Movie")
        st.markdown(
            f"""
            <div class="movie-card">
                <h3>{source_movie.get('title', 'Unknown')}</h3>
                <div class="meta">Genres: {source_movie.get('genres', 'N/A')} | Rating: {source_movie.get('vote_average', 'N/A')} | Popularity: {source_movie.get('popularity', 'N/A')}</div>
                <div class="tagline">{source_movie.get('tagline', '')}</div>
                <div>{str(source_movie.get('overview', 'No overview available.'))[:500]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Recommended Movies")
        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="movie-card">
                    <h3>{row.get('title', 'Unknown')}</h3>
                    <div class="meta">Genres: {row.get('genres', 'N/A')} | Rating: {row.get('vote_average', 'N/A')} | Popularity: {row.get('popularity', 'N/A')} | Similarity: {row.get('similarity_score', 0):.3f}</div>
                    <div class="tagline">{row.get('tagline', '')}</div>
                    <div>{str(row.get('overview', 'No overview available.'))[:400]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download recommendations as CSV",
            recommendations.to_csv(index=False).encode("utf-8"),
            file_name="movie_recommendations.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Built with Streamlit • Recommendation engine: TF-IDF + cosine similarity")
