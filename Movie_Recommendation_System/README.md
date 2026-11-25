# Movie Recommender

A small Streamlit app that uses the saved TF-IDF model to recommend similar movies. It runs locally and does not need a TMDB API key or a separate backend.

## Run locally (Windows)

Use Python 3.11 for the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Keep `df.pkl` and `tfidf_matrix.pkl` beside `app.py`. The other model artifacts and `movies_metadata.csv` are not needed by this app.

## Deploy on Streamlit Community Cloud

Push `app.py`, `requirements.txt`, `df.pkl`, and `tfidf_matrix.pkl` to a GitHub repository. In Streamlit Community Cloud, select that repository and set the app file to `app.py`; choose Python 3.11 in the app's advanced settings. The model files must be committed to the repository so the deployed app can load them.
