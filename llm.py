"""
TODO: This is the file you should edit.

get_recommendation() is called once per request with the user's input.
It should return a dict with keys "tmdb_id" and "description".

IMPORTANT: Do NOT hard-code your API key in this file. The grader will supply
its own OLLAMA_API_KEY environment variable when running your submission. Your
code must read it from the environment (os.environ or os.getenv), not from a
string literal in the source.
"""

import json
import os
import time
import argparse
import re
from functools import lru_cache
from difflib import SequenceMatcher, get_close_matches

import ollama
import pandas as pd


MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TIME_BUDGET_SECONDS = 19.0
EMERGENCY_TIME_THRESHOLD = 18.5

@lru_cache(maxsize=1)
def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Defensive cleanup in case some columns are missing
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
    "sad": ["not sad", "don't want sad", "dont want sad", "nothing depressing", "no sad movie"],
    "slow": ["not slow", "don't want slow", "dont want slow", "fast-paced only"],
    "violent": ["not violent", "don't want violence", "dont want violence", "no violence"],
}

COMMON_TYPO_FIXES = {
    "marval": "marvel",
    "advengers": "avengers",
    "avengerz": "avengers",
    "scifi": "sci-fi",
    "romcom": "romance comedy",
    "funy": "funny",
    "thirller": "thriller",
    "animtion": "animation",
    "drmaa": "drama",
    "comedyy": "comedy",
    "pixarrr": "pixar",
}

SERIES_KEYWORDS = {
    "avengers": ["avengers", "marvel", "mcu"],
    "toy story": ["toy story"],
    "harry potter": ["harry potter"],
    "lord of the rings": ["lord of the rings", "lotr"],
    "star wars": ["star wars"],
    "batman": ["batman", "dark knight"],
    "spider-man": ["spider-man", "spiderman"],
    "john wick": ["john wick"],
    "mission impossible": ["mission impossible"],
    "guardians": ["guardians of the galaxy", "guardians"],
}

GENRE_VOCAB = list(GENRE_HINTS.keys()) + [
    "superhero",
    "marvel",
    "pixar",
    "disney",
    "mind-bending",
    "feel-good",
    "family",
    "mystery",
    "avengers",
    "romance",
    "comedy",
    "thriller",
    "animation",
]


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s\-:]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def norm_title(s: str) -> str:
    return normalize_text(s)


def apply_typo_fixes(text: str) -> str:
    text = normalize_text(text)
    words = text.split()
    fixed = []

    vocab = list(COMMON_TYPO_FIXES.keys()) + GENRE_VOCAB

    for w in words:
        if w in COMMON_TYPO_FIXES:
            fixed.append(COMMON_TYPO_FIXES[w])
            continue

        match = get_close_matches(w, vocab, n=1, cutoff=0.88)
        if match:
            candidate = match[0]
            fixed.append(COMMON_TYPO_FIXES.get(candidate, candidate))
        else:
            fixed.append(w)

    return " ".join(fixed)


def token_set(s: str) -> set[str]:
    return set(normalize_text(s).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def get_client():
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OLLAMA_API_KEY environment variable.")

    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def extract_json_object(text: str) -> dict | None:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def title_matches_history(movie_title: str, history_entry: str) -> bool:
    movie_n = normalize_text(movie_title)
    hist_n = normalize_text(history_entry)

    if not hist_n:
        return False

    # Exact title match
    if movie_n == hist_n:
        return True

    # Strong substring match
    if hist_n in movie_n or movie_n in hist_n:
        if len(hist_n) >= 6 or len(movie_n) >= 6:
            return True

    # Token overlap
    movie_tokens = token_set(movie_n)
    hist_tokens = token_set(hist_n)
    if movie_tokens and hist_tokens:
        overlap = len(movie_tokens & hist_tokens) / max(1, len(hist_tokens))
        if overlap >= 0.8:
            return True

    # Fuzzy similarity
    if similarity(movie_n, hist_n) >= 0.86:
        return True

    # Franchise / series blocking
    for _, keywords in SERIES_KEYWORDS.items():
        hist_has_series = any(k in hist_n for k in keywords)
        movie_has_series = any(k in movie_n for k in keywords)
        if hist_has_series and movie_has_series:
            return True

    return False


def filter_seen_movies(movies: pd.DataFrame, history: list[str], history_ids: list[int]) -> pd.DataFrame:
    seen_ids = set()
    for x in history_ids:
        try:
            if pd.notna(x):
                seen_ids.add(int(x))
        except Exception:
            pass

    cleaned_history = [apply_typo_fixes(x) for x in history if str(x).strip()]

    filtered = movies.copy()

    if "tmdb_id" in filtered.columns and seen_ids:
        filtered = filtered[~filtered["tmdb_id"].isin(seen_ids)]

    def is_seen_title(title: str) -> bool:
        for h in cleaned_history:
            if title_matches_history(title, h):
                return True
        return False

    filtered = filtered[~filtered["title"].apply(is_seen_title)]
    return filtered


def score_movie(preferences: str, row) -> float:
    text = apply_typo_fixes(preferences)
    genres = str(row.genres).lower() if pd.notna(row.genres) else ""
    overview = str(row.overview).lower() if pd.notna(row.overview) else ""
    title = str(row.title).lower() if pd.notna(row.title) else ""

    score = 0.0

    # Positive preference matching
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
        if 1980 <= year <= 2010:
            score += 1.5

    # Mood / pace
    if any(word in text for word in ["fun", "easy", "light", "comfort", "feel-good", "lighthearted"]):
        if "comedy" in genres or "animation" in genres:
            score += 2.0

    if any(word in text for word in ["emotional", "moving", "touching", "heartfelt"]):
        if "drama" in genres or "romance" in genres:
            score += 2.0

    if any(word in text for word in ["mind-bending", "thought-provoking", "smart", "brainy"]):
        if "science fiction" in genres or "thriller" in genres:
            score += 2.0

    if any(word in text for word in ["fast", "fast-paced", "exciting", "intense"]):
        if "action" in genres or "thriller" in genres:
            score += 2.0

    if "marvel" in text or "avengers" in text:
        if "marvel" in overview or "avengers" in title or "superhero" in genres or "action" in genres:
            score += 2.0

    # Negative preference penalties
    if any(phrase in text for phrase in NEGATIVE_HINTS["horror"]):
        if "horror" in genres:
            score -= 5.0

    if any(phrase in text for phrase in NEGATIVE_HINTS["violent"]):
        if "action" in genres or "crime" in genres or "thriller" in genres:
            score -= 2.0

    if any(phrase in text for phrase in NEGATIVE_HINTS["sad"]):
        if "drama" in genres:
            score -= 1.5

    return score


def rank_candidates(preferences: str, movies: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    ranked = movies.copy()
    ranked["score"] = ranked.apply(lambda row: score_movie(preferences, row), axis=1)
    ranked = ranked.sort_values(by="score", ascending=False).head(top_k).copy()
    return ranked


def build_candidate_block(candidates: pd.DataFrame) -> str:
    lines = []
    for row in candidates.itertuples():
        overview = str(row.overview) if pd.notna(row.overview) else ""
        overview = overview.replace("\n", " ").strip()
        overview = overview[:180]

        year = getattr(row, "year", "")
        genres = str(row.genres) if pd.notna(row.genres) else ""
        vote = getattr(row, "vote_average", "")

        lines.append(
            f'- tmdb_id={int(row.tmdb_id)} | "{row.title}" ({year}) | '
            f'genres: {genres} | rating: {vote} | overview: {overview}'
        )
    return "\n".join(lines)


def choose_movie_with_llm(preferences: str, history: list[str], candidates: pd.DataFrame) -> dict | None:
    history_text = ", ".join(f'"{h}"' for h in history) if history else "none"
    candidate_block = build_candidate_block(candidates)
    fixed_preferences = apply_typo_fixes(preferences)

    prompt = f"""You are a movie recommendation agent.

User preferences:
{fixed_preferences}

Movies already watched:
{history_text}

Choose exactly ONE movie from the candidate list below.
Do not recommend anything already watched.
Only choose a tmdb_id that appears in the candidate list.

Return ONLY valid JSON in this exact format:
{{
  "tmdb_id": 123,
  "reason": "one short sentence"
}}

Candidate movies:
{candidate_block}
"""

    try:
        client = get_client()
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        parsed = extract_json_object(response.message.content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def clean_description(text: str) -> str:
    if not text:
        return "A strong pick with broad appeal, memorable storytelling, and a style that fits what you're looking for."

    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()

    if len(text) > 500:
        text = text[:497].rstrip() + "..."

    return text


def fallback_description(preferences: str, chosen_row) -> str:
    prefs = apply_typo_fixes(preferences)
    title = str(chosen_row.title)
    genres = str(chosen_row.genres) if pd.notna(chosen_row.genres) else "varied genres"
    overview = str(chosen_row.overview) if pd.notna(chosen_row.overview) else ""
    overview = overview.replace("\n", " ").strip()

    if len(overview) > 180:
        overview = overview[:177].rstrip() + "..."

    angle = "a polished, easy-to-enjoy pick"
    if any(word in prefs for word in ["fun", "funny", "light", "easy", "feel-good", "lighthearted"]):
        angle = "a fun, easy watch with strong crowd-pleasing energy"
    elif any(word in prefs for word in ["mind-bending", "smart", "thought-provoking", "sci-fi", "science fiction"]):
        angle = "a satisfying choice if you want something clever and immersive"
    elif any(word in prefs for word in ["emotional", "moving", "heartfelt", "touching"]):
        angle = "a strong pick if you want something emotionally engaging"
    elif any(word in prefs for word in ["exciting", "action", "fast-paced", "intense"]):
        angle = "a great match if you want something high-energy and gripping"
    elif "marvel" in prefs or "avengers" in prefs:
        angle = "a solid superhero pick with big-scale entertainment value"

    desc = f"{title} is {angle}. It blends {genres} with a clear hook—{overview}"
    return clean_description(desc)


def choose_best_fallback(candidates: pd.DataFrame):
    if len(candidates) == 0:
        raise RuntimeError("No candidate movies available after filtering.")
    return candidates.iloc[0]


def movie_conflicts_with_history(chosen_row, history: list[str], history_ids: list[int]) -> bool:
    title = str(chosen_row.title)

    try:
        chosen_id = int(chosen_row.tmdb_id)
    except Exception:
        chosen_id = None

    for hid in history_ids:
        try:
            if chosen_id is not None and int(hid) == chosen_id:
                return True
        except Exception:
            pass

    for h in history:
        fixed_h = apply_typo_fixes(h)
        if title_matches_history(title, fixed_h):
            return True

    return False

def time_exceeded(start_time: float, threshold: float = EMERGENCY_TIME_THRESHOLD) -> bool:
    return (time.perf_counter() - start_time) >= threshold


def emergency_random_pick(candidates: pd.DataFrame) -> dict:
    if len(candidates) == 0:
        raise RuntimeError("No candidates available for emergency fallback.")

    chosen = candidates.sample(n=1).iloc[0]

    title = str(chosen.title)
    genres = str(chosen.genres) if pd.notna(chosen.genres) else "mixed genres"
    overview = str(chosen.overview) if pd.notna(chosen.overview) else ""
    overview = overview.replace("\n", " ").strip()

    if len(overview) > 120:
        overview = overview[:117].rstrip() + "..."

    description = (
        f"I took a quick wildcard pick to keep this recommendation fast and safe under the time limit. "
        f"{title} stands out as a broadly watchable option in {genres}. "
        f"{overview} "
        f"If you want something more tailored, try a slightly simpler prompt and I can aim more precisely."
    )

    return {
        "tmdb_id": int(chosen.tmdb_id),
        "description": clean_description(description),
    }

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    movies = load_movies()

    # Step 1: filter out seen / franchise-related movies
    candidates = filter_seen_movies(movies, history, history_ids)
    if len(candidates) == 0:
        candidates = movies.copy()

    # Step 2: rank candidates quickly
    ranked = rank_candidates(preferences, candidates, top_k=8)

    # Step 3: ask LLM to choose only from a small, safe candidate list
    llm_choice = choose_movie_with_llm(preferences, history, ranked)

    chosen_row = None
    valid_ids = set(int(x) for x in ranked["tmdb_id"].tolist())

    if llm_choice and "tmdb_id" in llm_choice:
        try:
            chosen_id = int(llm_choice["tmdb_id"])
            if chosen_id in valid_ids:
                match = ranked[ranked["tmdb_id"] == chosen_id]
                if len(match) > 0:
                    chosen_row = match.iloc[0]
        except Exception:
            chosen_row = None

    if chosen_row is None:
        chosen_row = choose_best_fallback(ranked)

    # Step 4: last guardrail against seen / series conflicts
    if movie_conflicts_with_history(chosen_row, history, history_ids):
        for _, row in ranked.iterrows():
            if not movie_conflicts_with_history(row, history, history_ids):
                chosen_row = row
                break

    # Step 5: deterministic description for speed
    description = fallback_description(preferences, chosen_row)

    result = {
        "tmdb_id": int(chosen_row.tmdb_id),
        "description": clean_description(description),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history",
        type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
    parser.add_argument(
        "--history_ids",
        type=str,
        help='Comma-separated TMDB ids. Example: "299536,862"',
    )
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    print("For watch history, enter comma-separated movie titles (or leave blank).")

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history_ids_raw = (
        args.history_ids.strip()
        if args.history_ids and args.history_ids.strip()
        else input("Watch history TMDB ids (optional): ").strip()
    )

    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    history_ids = []
    if history_ids_raw:
        for x in history_ids_raw.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                history_ids.append(int(x))
            except ValueError:
                pass

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history, history_ids)
    print(result)
    elapsed = time.perf_counter() - start

    print(f"\nServed in {elapsed:.2f}s")
