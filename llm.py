"""
Agentic Movie Recommender
Modified to return movie details and concise descriptions for UI display.
"""

import json
import os
import time
import argparse
import re
import concurrent.futures
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
    """Converts text to lowercase and removes special characters, but keeps hyphens."""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)  # 這裡加了 \- 保留連字號
    return re.sub(r"\s+", " ", s)


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
            candidate = m.group(1).strip()
            signals["director"] = candidate
            signals["required_terms"].add(candidate)
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
    """
    Scores a movie based on n-gram overlap, STRICT year matching, adult themes, and nationality mapping.
    """
    prompt_clean = signals.get("text", "")
    prompt_normalized = prompt_clean.replace(" s", "s").replace("'s", "s")
    
    blob = str(row.get('_search_blob', ''))
    
    # [CRITICAL FIX 1]: Create a string combining ALL column values in the row to catch hidden data
    # like 'production_countries' or 'original_language' that might not be in the search blob.
    full_row_text = " ".join([str(val).lower() for val in row.values])
    
    score = 0.0
    
    # ==========================================
    # 1. BULLETPROOF Year & Decade Parsing
    # ==========================================
    movie_year_str = str(row.get("year", ""))
    movie_year = 0
    if movie_year_str.replace('.', '').isdigit():
        movie_year = int(float(movie_year_str))

    has_year_constraint = False
    year_matched = False

    if movie_year > 0:
        range_match = re.search(r"\b(19\d{2}|20\d{2})\s*(?:-|to|and|until)\s*(19\d{2}|20\d{2})\b", prompt_normalized)
        decade_match = re.search(r"\b(early|mid|late)?\s*(19|20)?(\d)0s\b", prompt_normalized)
        exact_match = re.search(r"\b(19\d{2}|20\d{2})\b", prompt_normalized)

        if range_match:
            has_year_constraint = True
            start, end = int(range_match.group(1)), int(range_match.group(2))
            start, end = min(start, end), max(start, end)
            if start <= movie_year <= end:
                year_matched = True
                score += 100.0 
                
        elif decade_match:
            has_year_constraint = True
            prefix = decade_match.group(1) or ""
            century = decade_match.group(2)
            decade_digit = decade_match.group(3)
            
            base_century = century if century else ("20" if int(decade_digit) < 5 else "19")
            base_year = int(f"{base_century}{decade_digit}0")

            if prefix == "early":
                year_matched = (base_year <= movie_year <= base_year + 3)
            elif prefix == "mid":
                year_matched = (base_year + 3 <= movie_year <= base_year + 6)
            elif prefix == "late":
                year_matched = (base_year + 6 <= movie_year <= base_year + 9)
            else:
                year_matched = (base_year <= movie_year <= base_year + 9)

            if year_matched:
                score += 100.0
                
        elif exact_match:
            has_year_constraint = True
            if int(exact_match.group(1)) == movie_year:
                year_matched = True
                score += 100.0

    # ⭐ THE KILL SWITCH ⭐
    if has_year_constraint and not year_matched:
        score -= 2000.0  

    # ==========================================
    # 2. Nationality & Culture Mapping (FIXED)
    # ==========================================
    country_synonyms = {
        "chinese": ["china", "hong kong", "taiwan", "mandarin", "cantonese", "beijing"],
        "japanese": ["japan", "tokyo", "anime", "kyoto"],
        "korean": ["korea", "seoul", "busan"],
        "french": ["france", "paris", "french"],
        "british": ["uk", "united kingdom", "london", "britain", "england"],
        "english": ["uk", "united kingdom", "london", "britain", "england"],
        "indian": ["india", "bollywood", "hindi", "mumbai"],
        "spanish": ["spain", "mexico", "madrid", "barcelona"],
        "italian": ["italy", "rome", "milan"],
        "german": ["germany", "berlin"]
    }
    
    for nat, locations in country_synonyms.items():
        if nat in prompt_normalized:
            for loc in locations:
                # Search in the FULL row text, not just the blob
                if loc in full_row_text:
                    score += 150.0  # Massive boost to ensure it surfaces

    # ==========================================
    # 3. Adult & Steamy Themes Mapping
    # ==========================================
    genre_synonyms = {
        "funny": "comedy", "hilarious": "comedy",
        "scary": "horror", "creepy": "horror",
        "sad": "drama", "emotional": "drama",
        "exciting": "action", "fast": "action",
        "love": "romance", "romantic": "romance",
        "sci fi": "science fiction", "space": "science fiction",
        "porn": "romance", "hot": "romance", "sexy": "romance", 
        "steamy": "romance", "erotic": "thriller"
    }
    
    adult_keywords = ["porn", "hot", "sexy", "steamy", "erotic"]
    if any(word in prompt_normalized for word in adult_keywords):
        target_vibes = ["affair", "seduction", "erotic", "sex", "desire", "passion", "sensual"]
        for vibe in target_vibes:
            if vibe in blob:
                score += 15.0  

    # ==========================================
    # 4. Standard N-gram & Keyword Boosting
    # ==========================================
    tokens = prompt_normalized.split()
    extended_tokens = list(tokens)
    for word, genre in genre_synonyms.items():
        if word in tokens:
            extended_tokens.append(genre)

    # 1-gram
    for t in extended_tokens:
        if len(t) > 3 and t in blob:
            score += 1.0
            
    # 2-gram (Actor names like "Emma Stone")
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if len(bigram) > 4 and bigram in blob:
            score += 80.0  
            
    # 3-gram
    for i in range(len(tokens) - 2):
        trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        if len(trigram) > 6 and trigram in blob:
            score += 120.0 

    # Base weights
    vote = float(row.get("vote_average", 0.0)) if pd.notna(row.get("vote_average")) else 0.0
    popularity = float(row.get("popularity", 0.0)) if pd.notna(row.get("popularity")) else 0.0
    
    score += vote * 0.5
    # The popularity acts as a perfect tie-breaker. If Emma and Adam both get +80, 
    # Cruella's higher popularity will make it rank higher than Hotel Transylvania!
    score += popularity * 0.01
    
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
        
        cast_info = getattr(row, "top_cast", getattr(row, "cast", ""))
        cast_str = str(cast_info) if pd.notna(cast_info) else "Unknown"

        lines.append(
            f'- tmdb_id={int(row.tmdb_id)} | "{row.title}" ({year}) | '
            f'cast: {cast_str} | genres: {genres} | rating: {vote} | overview: {overview}'
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
The candidates are already sorted by how well they match the user's prompt and their overall popularity.

Return ONLY valid JSON in this exact format:
{{
  "tmdb_id": 123,
  "description": "Your 2-3 sentence explanation here following the critical rules."
}}

Candidate movies (ordered by best match and popularity):
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
    """Cleans the text for display, removing quotes and fixing whitespace."""
    if not text:
        return "A great choice that matches your mood and specific interests."

    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()

    return text


def fallback_description(preferences: str, chosen_row) -> str:
    """Generates a concise 2-sentence description for fallback scenarios."""
    signals = extract_preference_signals(preferences)
    title = str(chosen_row.title)
    genres = str(chosen_row.genres) if pd.notna(chosen_row.genres) else "varied genres"

    if signals["franchise"] == "marvel":
        reason = "Since you're looking for Marvel, this fits perfectly into that universe with high-stakes action."
    elif signals["actor"]:
        reason = f"This is a top-tier choice featuring {signals['actor']}, exactly as you requested."
    elif signals["weather"] == "rain":
        reason = "This film captures the rainy, atmospheric mood from your prompt perfectly."
    else:
        reason = f"This {genres} pick aligns with the mood and details found in your request."

    return f"{reason} It’s an engaging experience that makes for a great watch tonight."


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

    return {
        "tmdb_id": int(chosen.tmdb_id),
        "title": str(chosen.title),
        "year": str(getattr(chosen, "year", "")),
        "description": "I picked a fast backup option to stay within the time limit. This highly-rated film fits the general style of your prompt.",
    }


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    start_time = time.perf_counter()
    movies = load_movies()

    candidates = filter_seen_movies(movies, history, history_ids)
    if len(candidates) == 0:
        candidates = movies.copy()

    ranked = rank_candidates(preferences, candidates, top_k=8)
    if len(ranked) == 0:
        ranked = candidates.head(8).copy()

    llm_choice = choose_movie_with_llm(preferences, history, ranked, start_time)

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
    # Construct the final response with name and year
    result = {
        "tmdb_id": int(chosen_row.tmdb_id),
        "title": str(chosen_row.title),
        "year": str(getattr(chosen_row, "year", "")),
        "description": clean_description(llm_choice.get("description")) if llm_choice else fallback_description(preferences, chosen_row),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local movie recommendation test.")
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    parser.add_argument("--history_ids", type=str)
    args = parser.parse_args()

    preferences = args.preferences.strip() if args.preferences else input("What are you in the mood for?: ").strip()
    history_raw = args.history.strip() if args.history else input("Watch History (optional, comma-separated): ").strip()
    history_ids_raw = args.history_ids.strip() if args.history_ids else input("Watch History IDs (optional, comma-separated): ").strip()

    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []
    history_ids = [int(x.strip()) for x in history_ids_raw.split(",") if x.strip().isdigit()]

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history, history_ids)
    
    # Fancy local print output
    print(f"Recommended: {result['title']} ({result['year']})")
    print(f"Description: {result['description']}")
    
    elapsed = time.perf_counter() - start
    print(f"\nServed in {elapsed:.2f}s")
