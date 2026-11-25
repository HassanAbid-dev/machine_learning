import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


MODEL_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = ("df.pkl", "tfidf_matrix.pkl")


@st.cache_resource
def load_model():
    missing_files = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing_files)
        )

    with (MODEL_DIR / "df.pkl").open("rb") as model_file:
        movies = pickle.load(model_file)
    with (MODEL_DIR / "tfidf_matrix.pkl").open("rb") as model_file:
        tfidf_matrix = pickle.load(model_file)

    if not isinstance(movies, pd.DataFrame) or "title" not in movies.columns:
        raise ValueError("df.pkl must contain a pandas DataFrame with a title column.")
    if tfidf_matrix.shape[0] != len(movies):
        raise ValueError("The TF-IDF matrix row count does not match df.pkl.")

    title_to_index = {}
    for index, title in enumerate(movies["title"]):
        if pd.notna(title):
            title_to_index.setdefault(str(title), index)

    if not title_to_index:
        raise ValueError("No movie titles were found in df.pkl.")

    return tfidf_matrix, title_to_index, movies["title"].tolist()


def get_recommendations(tfidf_matrix, movie_index, count):
    movie_vector = tfidf_matrix[movie_index]
    if movie_vector.nnz == 0:
        return []

    scores = (tfidf_matrix @ movie_vector.T).toarray().ravel()
    scores[movie_index] = -1
    best_indices = np.argsort(scores)[::-1][:count]
    return [
        (index, float(scores[index]))
        for index in best_indices
        if scores[index] > 0
    ]


st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.write("Choose a movie from your saved model and find similar titles.")

tfidf_matrix, title_to_index, titles_by_index = load_model()
movie_titles = list(title_to_index)

query = st.text_input("Search for a movie", placeholder="For example: Toy Story")
if not query.strip():
    st.caption(f"Your model contains {len(movie_titles):,} movie titles.")
    st.stop()

matches = [
    title for title in movie_titles if query.strip().casefold() in title.casefold()
][:100]
if not matches:
    st.warning("No matching movie found. Try another title.")
    st.stop()

selected_title = st.selectbox("Select a movie", matches)
recommendation_count = st.slider("Number of recommendations", 1, 20, 10)

if st.button("Get recommendations", type="primary"):
    movie_index = title_to_index[selected_title]
    recommendations = get_recommendations(
        tfidf_matrix, movie_index, recommendation_count
    )
    if not recommendations:
        st.info("This movie has no searchable description in the saved model.")
    else:
        st.subheader(f"Movies similar to {selected_title}")
        results = pd.DataFrame(
            [
                {
                    "Movie": str(titles_by_index[index]),
                    "Similarity": f"{score:.1%}",
                }
                for index, score in recommendations
            ]
        )
        st.dataframe(results, hide_index=True, use_container_width=True)
