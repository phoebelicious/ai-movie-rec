# Agentic Movie Recommender

## 1. Approach and Architecture
This project implements a hybrid Agentic Movie Recommender that combines a **high-speed deterministic retrieval engine** with a **Large Language Model (LLM)** for synthesis and personalized generation. 

Instead of feeding the entire database to the LLM (which risks hallucination and latency timeouts), I built a robust pipeline:
1. **Intelligent Pre-Filtering & Scoring (Custom Search Tool):** - **N-gram & Keyword Overlap:** Breaks down user prompts into unigrams, bigrams (for actor names), and trigrams to calculate a baseline match score.
   - **Bulletproof Regex Parsing (The Kill Switch):** Uses advanced Regular Expressions to extract precise decades (e.g., "early 2000s") or ranges (e.g., "2000-2010"). If a movie fails a strict temporal constraint, a "Kill Switch" applies a -2000 point penalty to guarantee it is filtered out.
   - **Semantic & Nationality Mapping:** Maps colloquial terms (e.g., "Chinese") to database equivalents ("China", "Hong Kong", "Mandarin") by scanning the entire dataset row, and handles implicit/adult themes by mapping them to appropriate TMDB underlying tags (e.g., "seduction", "thriller").
2. **LLM Synthesis & Anti-Hallucination:**
   - The top 8 candidates are passed to `gemma4:31b-cloud` in JSON format alongside the user's watch history.
   - **Strict Prompt Engineering:** The prompt includes a specific "Missing Actors Protocol" and anti-hallucination constraints, forcing the LLM to honestly acknowledge when no single movie contains all requested actors, defaulting to the highest-popularity alternative instead of fabricating facts.
3. **Time-Budget Management:** - Strict `time.perf_counter()` checks (18.5s threshold). If the LLM lags, the system safely aborts the API call and triggers an elegant fallback mechanism to guarantee a response within the 20-second hard limit.

## 2. Evaluation Strategy
To ensure the robustness of the system, I evaluated the agent across several rigorous test cases:
- **Temporal Edge Cases:** Tested phrases like "late 90s", "between 2005 and 2010", and "early 2000s" to ensure the regex parser correctly calculated the mathematical windows and applied the Kill Switch.
- **Hallucination Traps:** Prompted the system with "a movie starring both Emma Stone and Adam Sandler" (who have no movies together in the dataset). I verified that the system gracefully degraded to recommending the most popular movie of just one of the actors (*Cruella*) while explicitly explaining the compromise in the description.
- **Latency Limits:** Profiled the pipeline to ensure the local retrieval phase takes < 0.1s, reserving the remaining 19s entirely for the Ollama API call.

## 3. Brief Guide to the Code
- **`llm.py`:** The core engine.
  - `score_movie()`: The algorithmic heart of the retrieval system (handles N-grams, regex, and implicit mapping).
  - `filter_seen_movies()`: Excludes watch history using exact ID mapping and fuzzy title matching.
  - `get_recommendation()`: The main entry point that chains filtering, scoring, LLM calling, and fallback safety nets.
- **`app.py`:** A Streamlit-based web interface for an interactive, premium user experience.
- **`test.py`:** The grading script used to validate schema compliance and timeout limits.

### How to Run (CLI)
Ensure your `OLLAMA_API_KEY` is set in your environment variables.
```bash
python llm.py --preferences "a funny movie with Emma Stone" --history "The Help"
