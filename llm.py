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

    # Defensive cleanup
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
    if "tmdb_id" not in df.columns and "id" in df.columns:
        df["tmdb_id"] = df["id"]

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
    "raincouver": "rain vancouver sucks",
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

FRANCHISE_TERMS = {
    "marvel": [
        "marvel", "mcu", "avengers", "iron man", "captain america",
        "thor", "black widow", "hulk", "guardians of the galaxy",
        "doctor strange", "spider-man", "ant-man", "wakanda", "black panther"
    ],
    "dc": [
        "dc", "batman", "superman", "wonder woman", "justice league",
        "joker", "aquaman", "flash"
    ],
    "pixar": ["pixar", "toy story", "inside out", "up", "coco", "monsters inc"],
    "disney": ["disney", "frozen", "moana", "encanto", "tangled"],
}

HOLIDAY_TERMS = {
    "christmas": ["christmas", "xmas", "holiday", "holidays", "winter holiday"],
    "halloween": ["halloween", "spooky season"],
    "valentine": ["valentine", "valentines", "romantic holiday"],
    "new year": ["new year", "new years", "nye"],
    "thanksgiving": ["thanksgiving"],
}

SEASON_TERMS = {
    "summer": ["summer", "sunny", "beach", "vacation"],
    "fall": ["fall", "autumn"],
    "winter": ["winter", "snow", "cold"],
    "spring": ["spring", "bloom"],
}

WEATHER_TERMS = {
    "rain": ["rain", "rainy", "storm", "drizzle"],
    "snow": ["snow", "snowy", "blizzard"],
    "sun": ["sun", "sunny", "warm weather"],
}

SITUATION_TERMS = {
    "breakup": ["break up", "breakup", "broke up", "heartbroken"],
    "celebration": ["celebration", "celebrate", "party", "birthday", "graduation", "wedding"],
    "comfort": ["comfort", "comforting", "cozy", "easy watch"],
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


def time_exceeded(start_time: float, threshold: float = EMERGENCY_TIME_THRESHOLD) -> bool:
    return (time.perf_counter() - start_time) >= threshold


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


def get_col_value(row, possible_cols: list[str]) -> str:
    for col in possible_cols:
        if hasattr(row, col):
            value = getattr(row, col)
            if pd.notna(value):
                return str(value)
    return ""


def movie_metadata_blob(row) -> str:
    fields = [
        get_col_value(row, ["title"]),
        get_col_value(row, ["genres"]),
        get_col_value(row, ["overview"]),
        get_col_value(row, ["keywords", "keyword", "tags"]),
        get_col_value(row, ["cast", "actors", "actor", "top_cast"]),
        get_col_value(row, ["director", "directors"]),
        get_col_value(row, ["crew"]),
        get_col_value(row, ["production_companies", "studio"]),
    ]
    return normalize_text(" ".join(fields))


def row_matches_any_term(row, terms: list[str]) -> bool:
    blob = movie_metadata_blob(row)
    return any(normalize_text(term) in blob for term in terms)


def dataframe_hard_filter(movies: pd.DataFrame, matcher) -> pd.DataFrame:
    mask = movies.apply(lambda row: matcher(row), axis=1)
    filtered = movies[mask].copy()
    return filtered


def title_matches_history(movie_title: str, history_entry: str) -> bool:
    movie_n = normalize_text(movie_title)
    hist_n = normalize_text(history_entry)

    if not hist_n:
        return False

    if movie_n == hist_n:
        return True

    if hist_n in movie_n or movie_n in hist_n:
        if len(hist_n) >= 6 or len(movie_n) >= 6:
            return True

    movie_tokens = token_set(movie_n)
    hist_tokens = token_set(hist_n)
    if movie_tokens and hist_tokens:
        overlap = len(movie_tokens & hist_tokens) / max(1, len(hist_tokens))
        if overlap >= 0.8:
            return True

    if similarity(movie_n, hist_n) >= 0.86:
        return True

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


def extract_preference_signals(preferences: str) -> dict:
    text = apply_typo_fixes(preferences)

    signals = {
        "text": text,
        "required_terms": set(),
        "preferred_genres": set(),
        "avoid_genres": set(),
        "franchise": None,
        "actor": None,
        "director": None,
        "holiday": None,
        "season": None,
        "weather": None,
        "mood": None,
        "situation": [],
        "location_words": [],
        "emotion_words": [],
    }

    for genre, hints in GENRE_HINTS.items():
        if any(h in text for h in hints):
            signals["preferred_genres"].add(genre)

    if any(phrase in text for phrase in NEGATIVE_HINTS["horror"]):
        signals["avoid_genres"].add("horror")

    for franchise, terms in FRANCHISE_TERMS.items():
        if any(t in text for t in terms):
            signals["franchise"] = franchise
            signals["required_terms"].update(terms[:4])
            break

    actor_patterns = [
        r"(?:starring|with|featuring|actor|actress)\s+([a-z][a-z\s\.\-]{2,40})",
        r"i want\s+([a-z][a-z\s\.\-]{2,40})\s+movie",
        r"i want a\s+([a-z][a-z\s\.\-]{2,40})\s+movie",
        r"show me\s+([a-z][a-z\s\.\-]{2,40})",
        r"anything with\s+([a-z][a-z\s\.\-]{2,40})",
    ]
    for pat in actor_patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            if candidate not in {"funny movie", "romance movie", "marvel movie"}:
                signals["actor"] = candidate
                signals["required_terms"].add(candidate)
                break

    director_patterns = [
        r"(?:directed by|director)\s+([a-z][a-z\s\.\-]{2,40})",
        r"([a-z][a-z\s\.\-]{2,40})\s+directed",
        r"i want a\s+([a-z][a-z\s\.\-]{2,40})\s+film",
    ]
    for pat in director_patterns:
        m = re.search(pat, text)
        if m:
            signals["director"] = m.group(1).strip()
            signals["required_terms"].add(signals["director"])
            break

    for holiday, terms in HOLIDAY_TERMS.items():
        if any(t in text for t in terms):
            signals["holiday"] = holiday
            signals["required_terms"].update(terms[:3])
            break

    for season, terms in SEASON_TERMS.items():
        if any(t in text for t in terms):
            signals["season"] = season
            signals["required_terms"].update(terms[:2])
            break

    for weather, terms in WEATHER_TERMS.items():
        if any(t in text for t in terms):
            signals["weather"] = weather
            signals["required_terms"].update(terms[:2])
            break

    for situation, terms in SITUATION_TERMS.items():
        if any(t in text for t in terms):
            signals["situation"].append(situation)

    if "breakup" in signals["situation"]:
        signals["mood"] = "heartbroken"
        signals["preferred_genres"].update({"drama", "romance"})
        signals["avoid_genres"].add("horror")

    if "rain" in text or "rainy" in text:
        signals["weather"] = "rain"
    if "vancouver" in text:
        signals["location_words"].append("vancouver")
    if "sucks" in text or "miserable" in text or "bad mood" in text:
        signals["mood"] = "gloomy"

    useful_words = []
    stop = {
        "i", "want", "something", "anything", "movie", "a", "an", "the", "just",
        "went", "through", "from", "that", "with", "and", "or", "to", "of", "for",
        "it", "is", "was", "be", "my", "me", "show", "like"
    }
    for word in text.split():
        if word not in stop and len(word) >= 4:
            useful_words.append(word)

    signals["required_terms"].update(useful_words[:6])
    return signals


def apply_hard_constraints(movies: pd.DataFrame, signals: dict) -> pd.DataFrame:
    constrained = movies.copy()

    if signals["franchise"]:
        franchise_terms = FRANCHISE_TERMS.get(signals["franchise"], [])
        filtered = dataframe_hard_filter(
            constrained,
            lambda row: row_matches_any_term(row, franchise_terms)
        )
        if len(filtered) > 0:
            constrained = filtered

    if signals["actor"]:
        actor_name = normalize_text(signals["actor"])
        filtered = dataframe_hard_filter(
            constrained,
            lambda row: actor_name in movie_metadata_blob(row)
        )
        if len(filtered) > 0:
            constrained = filtered

    if signals["director"]:
        director_name = normalize_text(signals["director"])
        filtered = dataframe_hard_filter(
            constrained,
            lambda row: director_name in movie_metadata_blob(row)
        )
        if len(filtered) > 0:
            constrained = filtered

    if signals["holiday"]:
        terms = HOLIDAY_TERMS.get(signals["holiday"], [])
        filtered = dataframe_hard_filter(
            constrained,
            lambda row: row_matches_any_term(row, terms)
        )
        if len(filtered) > 0:
            constrained = filtered

    if signals["season"]:
        terms = SEASON_TERMS.get(signals["season"], [])
        filtered = dataframe_hard_filter(
            constrained,
            lambda row: row_matches_any_term(row, terms)
        )
        if len(filtered) > 0:
            constrained = filtered

    if signals["weather"]:
        terms = WEATHER_TERMS.get(signals["weather"], [])
        filtered = dataframe_hard_filter(
            constrained,
            lambda row: row_matches_any_term(row, terms)
        )
        if len(filtered) > 0:
            constrained = filtered

    return constrained


def score_movie(signals: dict, row) -> float:
    genres = str(row.genres).lower() if pd.notna(row.genres) else ""
    blob = movie_metadata_blob(row)

    score = 0.0

    term_hits = 0
    for term in signals["required_terms"]:
        if normalize_text(term) in blob:
            term_hits += 1
    score += term_hits * 2.0

    for genre in signals["preferred_genres"]:
        if genre in genres:
            score += 4.0

    for genre in signals["avoid_genres"]:
        if genre in genres:
            score -= 6.0

    if signals["franchise"] == "marvel":
        if row_matches_any_term(row, FRANCHISE_TERMS["marvel"]):
            score += 10.0
        else:
            score -= 8.0

    if signals["franchise"] == "pixar":
        if row_matches_any_term(row, FRANCHISE_TERMS["pixar"]):
            score += 8.0
        else:
            score -= 4.0

    if signals["franchise"] == "disney":
        if row_matches_any_term(row, FRANCHISE_TERMS["disney"]):
            score += 8.0
        else:
            score -= 4.0

    if signals["actor"]:
        if normalize_text(signals["actor"]) in blob:
            score += 12.0
        else:
            score -= 6.0

    if signals["director"]:
        if normalize_text(signals["director"]) in blob:
            score += 12.0
        else:
            score -= 6.0

    if "breakup" in signals["situation"]:
        if "romance" in genres or "drama" in genres:
            score += 5.0
        if any(x in blob for x in ["love", "relationship", "loss", "heart", "grief", "healing"]):
            score += 3.0
        if "comedy" in genres:
            score -= 2.0

    if "celebration" in signals["situation"]:
        if "comedy" in genres or "romance" in genres or "animation" in genres:
            score += 2.5
        if any(x in blob for x in ["party", "family", "wedding", "celebration", "joy"]):
            score += 2.0

    if signals["weather"] == "rain":
        if any(x in blob for x in ["rain", "storm", "melancholy", "moody", "lonely"]):
            score += 3.0

    if signals["season"] == "winter":
        if any(x in blob for x in ["winter", "snow", "holiday", "christmas"]):
            score += 3.0

    if signals["holiday"] == "christmas":
        if row_matches_any_term(row, HOLIDAY_TERMS["christmas"]):
            score += 5.0

    vote = float(row.vote_average) if pd.notna(row.vote_average) else 0.0
    pop = float(row.popularity) if pd.notna(row.popularity) else 0.0
    score += vote * 0.5
    score += pop * 0.01

    return score


def rank_candidates(preferences: str, movies: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    signals = extract_preference_signals(preferences)
    constrained = apply_hard_constraints(movies, signals)
    ranked = constrained.copy()

    ranked["score"] = ranked.apply(lambda row: score_movie(signals, row), axis=1)
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


def choose_movie_with_llm(preferences: str, history: list[str], candidates: pd.DataFrame, start_time: float) -> dict | None:
    if time_exceeded(start_time):
        return None

    history_text = ", ".join(f'"{h}"' for h in history) if history else "none"
    candidate_block = build_candidate_block(candidates)
    fixed_preferences = apply_typo_fixes(preferences)

    prompt = f"""You are a movie recommendation agent.

User preferences:
{fixed_preferences}

Movies already watched:
{history_text}

Choose exactly ONE movie from the candidate list below.
Prioritize the movie that best matches the user's exact request, mood, and named entities.
If the user mentions a franchise like Marvel, strongly prefer a movie clearly related to that franchise.
If the user mentions an actor or director, strongly prioritize a movie matching that person.
Only choose a tmdb_id that appears in the candidate list.
Do not recommend anything already watched.

Return ONLY valid JSON in this exact format:
{{
  "tmdb_id": 123,
  "reason": "one short sentence"
}}

Candidate movies:
{candidate_block}
"""

    if time_exceeded(start_time):
        return None

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
    signals = extract_preference_signals(preferences)

    title = str(chosen_row.title)
    genres = str(chosen_row.genres) if pd.notna(chosen_row.genres) else "varied genres"
    overview = str(chosen_row.overview) if pd.notna(chosen_row.overview) else ""
    overview = overview.replace("\n", " ").strip()

    if len(overview) > 130:
        overview = overview[:127].rstrip() + "..."

    prompt_link = ""
    if signals["actor"]:
        prompt_link = f"Since you asked for {signals['actor']}, this connects directly to that actor request."
    elif signals["director"]:
        prompt_link = f"Since you wanted something tied to {signals['director']}, this is aligned with that director cue."
    elif signals["franchise"] == "marvel":
        prompt_link = "Since you specifically asked for Marvel, this stays in that franchise lane instead of drifting into something random."
    elif signals["franchise"]:
        prompt_link = f"Since you mentioned {signals['franchise']}, this stays much closer to that lane."
    elif signals["holiday"]:
        prompt_link = f"You mentioned {signals['holiday']}, so this is aimed at that seasonal or celebratory vibe."
    elif signals["season"]:
        prompt_link = f"This fits the {signals['season']} mood you mentioned."
    elif signals["weather"] == "rain":
        prompt_link = "This matches the rainy, moody atmosphere in your prompt."
    elif "breakup" in signals["situation"]:
        prompt_link = "Because you mentioned a breakup, this leans more into emotional relevance than a generic pick."
    elif "celebration" in signals["situation"]:
        prompt_link = "Because your prompt suggests celebration energy, this aims for a more fitting upbeat or communal vibe."
    else:
        matched_terms = [t for t in list(signals["required_terms"])[:4] if normalize_text(t) in movie_metadata_blob(chosen_row)]
        if matched_terms:
            prompt_link = f"It connects back to your prompt through details like {', '.join(matched_terms[:3])}."
        else:
            prompt_link = "This is meant to feel tied to the mood and keywords in your prompt, not just broadly popular."

    desc = (
        f"{prompt_link} {title} blends {genres} with a setup that makes it a relevant choice here: "
        f"{overview}"
    )

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


def emergency_random_pick(candidates: pd.DataFrame) -> dict:
    if len(candidates) == 0:
        raise RuntimeError("No candidates available for emergency fallback.")

    pool = candidates.head(min(5, len(candidates)))
    chosen = pool.sample(n=1).iloc[0]

    title = str(chosen.title)
    genres = str(chosen.genres) if pd.notna(chosen.genres) else "mixed genres"
    overview = str(chosen.overview) if pd.notna(chosen.overview) else ""
    overview = overview.replace("\n", " ").strip()

    if len(overview) > 120:
        overview = overview[:117].rstrip() + "..."

    description = (
        f"I used a fast backup pick to stay within the time limit. "
        f"{title} is a flexible choice from the stronger candidates, with {genres} elements and a clear hook: {overview} "
        f"If you want something more specifically tailored, try a slightly simpler prompt."
    )

    return {
        "tmdb_id": int(chosen.tmdb_id),
        "description": clean_description(description),
    }


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    start_time = time.perf_counter()
    movies = load_movies()

    candidates = filter_seen_movies(movies, history, history_ids)
    if len(candidates) == 0:
        candidates = movies.copy()

    if time_exceeded(start_time):
        return emergency_random_pick(candidates)

    ranked = rank_candidates(preferences, candidates, top_k=8)
    if len(ranked) == 0:
        ranked = candidates.head(8).copy()

    if time_exceeded(start_time):
        return emergency_random_pick(ranked)

    llm_choice = choose_movie_with_llm(preferences, history, ranked, start_time)

    if time_exceeded(start_time):
        return emergency_random_pick(ranked)

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

    if movie_conflicts_with_history(chosen_row, history, history_ids):
        for _, row in ranked.iterrows():
            if not movie_conflicts_with_history(row, history, history_ids):
                chosen_row = row
                break

    if time_exceeded(start_time):
        return emergency_random_pick(ranked)

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
