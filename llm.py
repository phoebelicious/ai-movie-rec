import json
import os
import time
import argparse
import re
from functools import lru_cache

import ollama
import pandas as pd

MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")


@lru_cache(maxsize=1)
def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Safety cleanup
    if "overview" not in df.columns:
        df["overview"] = ""
    if "genres" not in df.columns:
        df["genres"] = ""
    if "popularity" not in df.columns:
        df["popularity"] = 0
    if "vote_average" not in df.columns:
        df["vote_average"] = 0
    if "year" not in df.columns:
        df["year"] = ""

    return df


GENRE_HINTS = {
    "action": ["action", "exciting", "adventure", "fight", "fast-paced", "superhero"],
    "comedy": ["comedy", "funny", "laugh", "light", "lighthearted", "feel-good"],
    "romance": ["romance", "romantic", "love", "relationship"],
    "horror": ["horror", "scary", "creepy", "terrifying"],
    "thriller": ["thriller", "tense", "suspense", "suspenseful"],
    "drama": ["drama", "emotional", "serious", "character-driven"],
    "science fiction": ["sci-fi", "science fiction", "space", "future", "technology", "mind-bending"],
    "animation": ["animated", "animation", "pixar", "disney", "family"],
    "fantasy": ["fantasy", "magic", "magical", "myth"],
    "crime": ["crime", "detective", "gangster", "heist", "mystery"],
}


NEGATIVE_HINTS = {
    "horror": ["not scary", "don't want scary", "dont want scary", "no horror", "avoid horror"],
    "sad": ["not sad", "don't want sad", "dont want sad", "no sad movie", "nothing depressing"],
    "slow": ["not slow", "don't want slow", "dont want slow", "fast-paced only"],
    "violent": ["not violent", "don't want violence", "dont want violence", "no violence"],
}


def norm_title(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def filter_seen_movies(movies: pd.DataFrame, history: list[str], history_ids: list[int]) -> pd.DataFrame:
    seen_ids = set()
    for x in history_ids:
        try:
            if pd.notna(x):
                seen_ids.add(int(x))
        except Exception:
            pass

    seen_titles = {norm_title(x) for x in history if str(x).strip()}

    filtered = movies.copy()

    if "tmdb_id" in filtered.columns and seen_ids:
        filtered = filtered[~filtered["tmdb_id"].isin(seen_ids)]

    filtered = filtered[~filtered["title"].apply(lambda x: norm_title(x) in seen_titles)]

    return filtered


def score_movie(preferences: str, row) -> float:
    text = preferences.lower()
    genres = str(row.genres).lower() if pd.notna(row.genres) else ""
    overview = str(row.overview).lower() if pd.notna(row.overview) else ""
    title = str(row.title).lower() if pd.notna(row.title) else ""

    score = 0.0

    # Positive matching
    for genre, hints in GENRE_HINTS.items():
        hint_hit = any(hint in text for hint in hints)
        if hint_hit:
            if genre in genres:
                score += 3.0
            if any(hint in overview for hint in hints):
                score += 1.0
            if any(hint in title for hint in hints):
                score += 0.5

    # Quality priors
    vote = float(row.vote_average) if pd.notna(row.vote_average) else 0.0
    pop = float(row.popularity) if pd.notna(row.popularity) else 0.0
    score += vote * 0.5
    score += pop * 0.01

    # Recency preference
    year = 0
    try:
        year = int(row.year)
    except Exception:
        year = 0

    if any(word in text for word in ["recent", "new", "newer", "modern"]):
        if year >= 2018:
            score += 2.0
        elif year >= 2010:
            score += 1.0

    if any(word in text for word in ["classic", "older", "old-school"]):
        if 1980 <= year <= 2010