import os
import pickle
from difflib import get_close_matches

import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(229, 9, 20, 0.10), transparent 22%),
            linear-gradient(180deg, #0b0b0f 0%, #111111 45%, #141414 100%);
        color: white;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    header, footer {
        visibility: hidden;
    }

    .hero-banner {
        position: relative;
        padding: 2.7rem 2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        background:
            linear-gradient(to right, rgba(0,0,0,0.90), rgba(0,0,0,0.50)),
            url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255,255,255,0.06);
        animation: fadeInUp 0.8s ease;
    }

    .brand {
        color: #E50914;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.6rem;
        max-width: 800px;
    }

    .hero-subtitle {
        color: #d1d5db;
        font-size: 1.05rem;
        max-width: 760px;
        margin-bottom: 1rem;
    }

    .chip-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .chip {
        background: rgba(229,9,20,0.14);
        border: 1px solid rgba(229,9,20,0.30);
        color: white;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin: 1rem 0 0.9rem 0;
        color: white;
        animation: fadeInUp 0.7s ease;
    }

    .search-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.2rem;
        margin-bottom: 1.4rem;
    }

    .selected-box {
        background: linear-gradient(180deg, #191919, #121212);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1.3rem;
        animation: fadeInUp 0.6s ease;
    }

    .movie-card {
        background: linear-gradient(180deg, #1b1b1b, #121212);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.28);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.6s ease;
        min-height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 18px 34px rgba(0,0,0,0.38);
    }

    .card-title {
        margin-top: 0.8rem;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.25rem;
        min-height: 2.6rem;
    }

    .card-meta {
        color: #bdbdbd;
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }

    .overview-text {
        color: #d8d8d8;
        font-size: 0.88rem;
        margin-top: 0.5rem;
        line-height: 1.3rem;
    }

    div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }

    .stTextInput input {
        background-color: #1a1a1a !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    .stSlider label, .stSelectbox label, .stTextInput label {
        color: white !important;
    }

    .stButton > button {
        background: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 0 18px rgba(229,9,20,0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
        padding: 0.75rem 1rem !important;
    }

    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 24px rgba(229,9,20,0.55);
    }

    .footer-text {
        text-align: center;
        color: #9f9f9f;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE CHECK
# =========================================================
REQUIRED_FILES = [
    "artifacts/df.pkl",
    "artifacts/tfidf_matrix.pkl",
    "artifacts/indices.pkl"
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_files:
    st.error(f"Missing files: {missing_files}")
    st.stop()


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_resource
def load_artifacts():
    with open("artifacts/df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("artifacts/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("artifacts/indices.pkl", "rb") as f:
        indices = pickle.load(f)

    df = df.reset_index(drop=True)
    return df, tfidf_matrix, indices


df, tfidf_matrix, indices = load_artifacts()


# =========================================================
# HELPERS
# =========================================================
def get_col(possible_names):
    for col in possible_names:
        if col in df.columns:
            return col
    return None


TITLE_COL = get_col(["title", "Title", "movie_title", "name"])
GENRE_COL = get_col(["genres", "genre", "listed_in"])
OVERVIEW_COL = get_col(["overview", "description", "summary", "tagline"])
RELEASE_COL = get_col(["release_date", "year", "release_year"])
POSTER_COL = get_col(["poster_url", "Poster", "image_url", "poster", "poster_path"])
RATING_COL = get_col(["vote_average", "rating", "imdb_rating"])
POPULARITY_COL = get_col(["popularity", "vote_count"])

if TITLE_COL is None:
    st.error("No title column found in df.pkl")
    st.stop()


def parse_genres(value):
    if pd.isna(value):
        return set()
    if isinstance(value, list):
        return {str(x).strip().lower() for x in value if str(x).strip()}
    return {x.strip().lower() for x in str(value).split(",") if x.strip()}


def extract_year(value):
    if pd.isna(value):
        return None
    value = str(value)
    if len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None


def normalize_series(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def fetch_poster_from_omdb(title):
    try:
        api_key = st.secrets["OMDB_API_KEY"]
    except Exception:
        return "https://via.placeholder.com/300x450.png?text=No+Poster"

    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("Response") == "True" and data.get("Poster") and data.get("Poster") != "N/A":
            return data["Poster"]
    except Exception:
        pass

    return "https://via.placeholder.com/300x450.png?text=No+Poster"


def get_poster_url(row):
    if POSTER_COL and pd.notna(row.get(POSTER_COL)):
        value = str(row[POSTER_COL]).strip()
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if POSTER_COL == "poster_path" and value:
            return f"https://image.tmdb.org/t/p/w500{value}"

    return fetch_poster_from_omdb(str(row.get(TITLE_COL, "")))


def get_year(row):
    if RELEASE_COL and pd.notna(row.get(RELEASE_COL)):
        return str(row[RELEASE_COL])[:4]
    return "N/A"


def get_genres_text(row):
    if GENRE_COL and pd.notna(row.get(GENRE_COL)):
        return str(row[GENRE_COL])
    return "Not available"


def get_overview(row):
    if OVERVIEW_COL and pd.notna(row.get(OVERVIEW_COL)):
        return str(row[OVERVIEW_COL])
    return "No overview available."


def get_rating(row):
    if RATING_COL and pd.notna(row.get(RATING_COL)):
        return str(row[RATING_COL])
    return "N/A"


@st.cache_data(show_spinner=False)
def get_suggestions(query, titles):
    if not query.strip():
        return []
    return get_close_matches(query, titles, n=10, cutoff=0.3)


def recommend_hybrid(movie_title, top_n=12):
    if movie_title not in indices:
        close = get_close_matches(movie_title, list(indices.keys()), n=1, cutoff=0.5)
        if not close:
            return pd.DataFrame()
        movie_title = close[0]

    idx = indices[movie_title]
    if idx >= len(df):
        return pd.DataFrame()

    base = df.copy()

    tfidf_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    base["tfidf_score"] = tfidf_scores

    query_row = df.iloc[idx]
    query_genres = parse_genres(query_row.get(GENRE_COL)) if GENRE_COL else set()
    query_year = extract_year(query_row.get(RELEASE_COL)) if RELEASE_COL else None

    def genre_overlap(row):
        if not GENRE_COL:
            return 0.0
        row_genres = parse_genres(row.get(GENRE_COL))
        if not query_genres or not row_genres:
            return 0.0
        inter = len(query_genres & row_genres)
        union = len(query_genres | row_genres)
        return inter / union if union else 0.0

    base["genre_score"] = base.apply(genre_overlap, axis=1)

    if RATING_COL:
        base["rating_score"] = normalize_series(base[RATING_COL])
    else:
        base["rating_score"] = 0.0

    if POPULARITY_COL:
        base["popularity_score"] = normalize_series(base[POPULARITY_COL])
    else:
        base["popularity_score"] = 0.0

    def year_closeness(row):
        if not RELEASE_COL:
            return 0.0
        row_year = extract_year(row.get(RELEASE_COL))
        if query_year is None or row_year is None:
            return 0.0
        diff = abs(query_year - row_year)
        return max(0.0, 1 - diff / 20)

    base["year_score"] = base.apply(year_closeness, axis=1)

    base["final_score"] = (
        0.60 * base["tfidf_score"] +
        0.15 * base["genre_score"] +
        0.10 * base["rating_score"] +
        0.10 * base["popularity_score"] +
        0.05 * base["year_score"]
    )

    base = base[base.index != idx]
    base = base.sort_values("final_score", ascending=False)
    base = base.drop_duplicates(subset=[TITLE_COL], keep="first")

    return base.head(top_n)


def render_movie_cards(rec_df, cols=4):
    if rec_df.empty:
        st.warning("No recommendations found.")
        return

    for start in range(0, len(rec_df), cols):
        row_cols = st.columns(cols)
        chunk = rec_df.iloc[start:start + cols]

        for col, (_, row) in zip(row_cols, chunk.iterrows()):
            with col:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                st.image(get_poster_url(row), width="stretch")
                st.markdown(
                    f"<div class='card-title'>{row.get(TITLE_COL, 'Untitled')}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='card-meta'>Year: {get_year(row)} • Rating: {get_rating(row)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='card-meta'>Genres: {get_genres_text(row)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='card-meta'>Match Score: {round(float(row.get('final_score', 0)), 4)}</div>",
                    unsafe_allow_html=True
                )
                overview = get_overview(row)
                short_overview = overview[:140] + "..." if len(overview) > 140 else overview
                st.markdown(
                    f"<div class='overview-text'>{short_overview}</div>",
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# UI
# =========================================================
st.markdown("""
<div class="hero-banner">
    <div class="brand">CINEMATCH</div>
    <div class="hero-title">Unlimited movie recommendations, personalized for you.</div>
    <div class="hero-subtitle">
        Search a movie you already love and instantly discover similar films using a hybrid recommendation engine with a cinematic Netflix-style interface.
    </div>
    <div class="chip-row">
        <div class="chip">TF-IDF Similarity</div>
        <div class="chip">Genre Matching</div>
        <div class="chip">Smart Ranking</div>
        <div class="chip">OMDb Posters</div>
    </div>
</div>
""", unsafe_allow_html=True)

all_titles = sorted(df[TITLE_COL].dropna().astype(str).unique().tolist())

st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Search a Movie</div>', unsafe_allow_html=True)

search_text = st.text_input(
    "Movie title",
    placeholder="Type Inception, Titanic, Avatar, Interstellar..."
)

suggestions = get_suggestions(search_text, all_titles)

if suggestions:
    selected_movie = st.selectbox("Suggestions", suggestions)
else:
    selected_movie = st.selectbox("Suggestions", all_titles[:100])

left_col, right_col = st.columns([1, 1])

with left_col:
    num_recommendations = st.slider("Number of recommendations", 4, 16, 8)

with right_col:
    recommend_clicked = st.button("Recommend Now")

st.markdown('</div>', unsafe_allow_html=True)

if selected_movie:
    selected_rows = df[df[TITLE_COL].astype(str) == str(selected_movie)]
    if not selected_rows.empty:
        selected_row = selected_rows.iloc[0]

        col_a, col_b = st.columns([1, 2.1], gap="large")

        with col_a:
            st.markdown('<div class="selected-box">', unsafe_allow_html=True)
            st.image(get_poster_url(selected_row), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="selected-box">', unsafe_allow_html=True)
            st.markdown(
                f"<div class='section-title' style='margin-top:0'>{selected_row[TITLE_COL]}</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Year:** {get_year(selected_row)}")
            st.markdown(f"**Genres:** {get_genres_text(selected_row)}")
            st.markdown(f"**Rating:** {get_rating(selected_row)}")
            st.markdown(f"**Overview:** {get_overview(selected_row)}")
            st.markdown('</div>', unsafe_allow_html=True)

if recommend_clicked:
    with st.spinner("Finding the best movies for you..."):
        rec_df = recommend_hybrid(selected_movie, top_n=num_recommendations)

    st.markdown('<div class="section-title">Recommended for You</div>', unsafe_allow_html=True)
    render_movie_cards(rec_df, cols=4)

st.markdown(
    '<div class="footer-text">Built with Streamlit • Hybrid Movie Recommendation System</div>',
    unsafe_allow_html=True
)