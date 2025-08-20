import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
QUERY_GEN_MODEL = "gpt-3.5-turbo"

RETRIEVER_MODEL = "JJTsao/fine-tuned_movie_retriever-bge-base-en-v1.5" 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_MOVIE_COLLECTION_NAME = "Movies_BGE_June"
QDRANT_TV_COLLECTION_NAME = "TV_Shows_BGE_June"

def get_output_path(media_type: str) -> str:
    output_dir = Path(__file__).resolve().parent.parent / "data" / "training_data"
    return os.path.join(output_dir, f"{media_type}_train.jsonl")

def get_cache_path(media_type: str) -> str:
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
    return os.path.join(cache_dir, f"{media_type}_vibe_cache.jsonl")

VIBE_PROMPT_TEMPLATE = """
You are a helpful AI assistant helping generate training data for a {media_type} recommendation system.

Given the following {media_type} overview, generate {num_queries} short natural-sounding queries a user might type if they are looking for a {media_type} *like this*. 

Focus on emotional tone, mood, or storytelling vibe — not just factual filters like genre or year. Use adjectives like "inspiring", "feel-good", "satirical", "heartbreaking", "suspenseful", "captivating", "though-provoking", etc.

For example: Mind-bending sci-fi {media_type}s with deep philosophical themes, or Dark and gritty dramas with character-driven narratives.

Each response should be a plain string on a new line. Do **not** include any numbering, bullet points, or dashes. 

---
{training_text}
---
Queries:
"""

METADATA_QUERY_TEMPLATES = [
    # Fuzzy Matching
    "Any {genre} {media_type}s starring {star_1}?",
    "I'm in the mood for {genre} {media_type}s from {decade} about {keyword_1}",
    "Something like a {genre} story dealing with {keyword_1} or {keyword_2}",
    "{genre} {media_type}s directed by {director}",

    # Medium Specificity
    "What are some must-watch {genre} {media_type}s from the {decade}s reflecting on {keyword_1} and {keyword_2}",
    "Any recommendations for top {genre} {media_type}s from {decade} featuring {star_2}?",
    "Memorable {genre} {media_type}s focused on {keyword_1} and grappling with {keyword_2}",
    "Know any good {media_type}s with both {star_1} and {star_2}?",
    "Can you recommend something where {star_1} struggles with {keyword_1} and {keyword_2}",
    "Could you suggest some intriguing {genre} {media_type}s by {director} that are built around {keyword_1} and {keyword_2}",
    "Looking for {media_type}s with {star_1} and {star_2} in the main cast and centered around {keyword_2}",

    # Precise Requirement
    "Iconic {genre} {media_type}s from the {decade}s touching on {keyword_1} and {keyword_2} with standout performances by {star_2}",
    "Please suggest {genre} {media_type}s portraying {keyword_1} and {keyword_2}, starring {star_1}, that came out in {year}",
    "Critically acclaimed {genre} {media_type}s directed by {director} exploring the themes of {keyword_1} and {keyword_2}",
]

if not OPENAI_API_KEY or not TMDB_API_KEY:
    raise ValueError("Missing API key(s).")
