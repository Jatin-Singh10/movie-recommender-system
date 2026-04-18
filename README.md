# Movie Recommendation System Deployment

This app deploys your content-based movie recommender using **Streamlit**.

## Files you need
Place these files inside an `artifacts/` folder next to `app.py`:
- `df.pkl`
- `tfidf_matrix.pkl`
- `indices.pkl`

Your project structure should look like this:

```bash
movie_recommender_app/
│── app.py
│── requirements.txt
│── README.md
└── artifacts/
    ├── df.pkl
    ├── tfidf_matrix.pkl
    └── indices.pkl
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload the app folder to GitHub.
2. Make sure the `artifacts/` folder is included.
3. Go to Streamlit Community Cloud.
4. Create a new app and connect your GitHub repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Important improvements over notebook version
- Removed Colab-specific file paths
- Added fuzzy title matching for better search experience
- Added a styled UI with cards and metrics
- Added CSV export for recommendations
- Added artifact validation and cleaner app structure

## Recommended next improvements
- Add poster images using the TMDB API
- Add autocomplete search dropdown
- Store artifacts with joblib / sparse formats for better performance
- Deploy with Docker or FastAPI later if you want API + frontend separation
