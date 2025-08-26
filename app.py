# app.py
import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from typing import Optional

# -------------------------
# CONFIG (read from st.secrets first, then env)
# -------------------------
# Prefer Streamlit secrets (works on Streamlit Cloud and locally if .streamlit/secrets.toml exists)
if "TMDB_API_KEY" in st.secrets:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
else:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"  # poster base

MOVIE_DICT_PATH = "movie_dict.pkl"    # adjust if needed
SIMILARITY_PATH = "similarity.pkl"    # adjust if needed

st.set_page_config(layout="wide", page_title="Movie Recommender")

# If API key not provided, show instructions and stop
if not TMDB_API_KEY:
    st.error(
        "TMDB API key not found.\n\n"
        "Locally: create `.streamlit/secrets.toml` with `TMDB_API_KEY = \"your_key\"` "
        "or set environment variable TMDB_API_KEY.\n\n"
        "On Streamlit Cloud: set the secret in the app settings (Secrets)."
    )
    st.stop()

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(movie_dict_path: str, similarity_path: str):
    with open(movie_dict_path, "rb") as f:
        movie_dict = pickle.load(f)
    movies_df = pd.DataFrame(movie_dict)

    with open(similarity_path, "rb") as f:
        sim = pickle.load(f)

    sim = np.array(sim)  # ensure numpy array
    return movies_df, sim

try:
    movies, similarity = load_data(MOVIE_DICT_PATH, SIMILARITY_PATH)
except FileNotFoundError as e:
    st.error(f"Could not find data files: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -------------------------
# UTILS
# -------------------------
def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ("movie_id", "id", "tmdb_id", "movieId", "MovieID")
    for c in candidates:
        if c in df.columns:
            return c
    return None

ID_COL = detect_id_column(movies)

@st.cache_data(show_spinner=False)
def fetch_poster(tmdb_id: Optional[int]) -> Optional[str]:
    """Fetch poster URL for a TMDB id. Cached for speed."""
    if tmdb_id is None:
        return None
    try:
        resp = requests.get(
            TMDB_BASE_URL.format(int(tmdb_id)),
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=6
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return TMDB_IMAGE_BASE + poster_path
        return None
    except Exception:
        return None

def recommend(movie_title: str, top_n: int = 5):
    """Return list of dicts: [{'title':..., 'poster':..., 'tmdb_id':...}, ...]"""
    matches = movies[movies["title"] == movie_title].index
    if len(matches) == 0:
        return []

    movie_index = matches[0]

    # validate similarity shape
    n_movies = len(movies)
    if similarity.ndim != 2 or similarity.shape[0] != n_movies or similarity.shape[1] != n_movies:
        raise ValueError(f"Similarity matrix shape {similarity.shape} doesn't match movies length {n_movies}.")

    distances = similarity[movie_index]
    scored = list(enumerate(distances))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    recommended_indices = []
    for idx, score in scored_sorted:
        if idx == movie_index:
            continue
        recommended_indices.append(idx)
        if len(recommended_indices) >= top_n:
            break

    results = []
    for idx in recommended_indices:
        row = movies.iloc[idx]
        title = row.get("title", "Unknown title")
        tmdb_id = None
        if ID_COL is not None and pd.notnull(row.get(ID_COL)):
            try:
                tmdb_id = int(row.get(ID_COL))
            except Exception:
                tmdb_id = None
        poster = fetch_poster(tmdb_id) if tmdb_id else None
        results.append({"title": title, "poster": poster, "tmdb_id": tmdb_id})
    return results

# -------------------------
# UI
# -------------------------
st.title("🎬 Movie Recommender System")

try:
    titles = movies['title'].astype(str).values
except Exception:
    st.error("No 'title' column found in movies DataFrame.")
    st.stop()

selected_movie_name = st.selectbox("Select Movie", titles)

if st.button("Recommend"):
    try:
        recs = recommend(selected_movie_name, top_n=5)
        if not recs:
            st.info("No recommendations found for this movie.")
        else:
            cols = st.columns(len(recs))
            for c, rec in zip(cols, recs):
                if rec["poster"]:
                    c.image(rec["poster"], use_container_width=True)
                    c.caption(f"{rec['title']}")
                else:
                    c.markdown(f"**{rec['title']}**")
                    c.write("Poster not available")
    except Exception as ex:
        st.error(f"Error: {ex}")
