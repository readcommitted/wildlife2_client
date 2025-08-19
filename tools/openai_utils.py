"""
openai_utils.py — Embedding & Summary Utility for Semantic Search
------------------------------------------------------------------

Provides functions for:

* Generating 1536-dimensional text embeddings using OpenAI's `text-embedding-3-small` model
* Generating natural language summaries with GPT-4 based on wildlife search inputs

Used for:
- Creating vector embeddings from structured text metadata (species, tags, location, behavior)
- Enabling semantic search within the Wildlife Vision System
- Providing contextual summaries about wildlife patterns during search

Requirements:
- `openai` package for API access
- `backoff` for retry logic
- `streamlit` for secure secrets management
- OPENAI_API_KEY must be set in `.streamlit/secrets.toml`

"""

from openai import OpenAI
import backoff
from typing import List
from config.settings import GPTMODEL, OPENAI_API_KEY, EMBED_MODEL


# --- OpenAI Client Setup ---
client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = EMBED_MODEL


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def get_embedding(text: str, model: str = EMBED_MODEL) -> List[float]:
    """
    Generates a normalized 1536-dimensional text embedding for a given input string.

    Args:
        text (str): Input text (e.g., species + tags + behavior).
        model (str): Embedding model to use (default: "text-embedding-3-small").

    Returns:
        List[float]: 1536-dimensional embedding vector.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("Text for embedding is empty")

    response = client.embeddings.create(
        model=model,
        input=[text]
    )

    return response.data[0].embedding


def summarize_wildlife_search(user_query: str) -> str:
    """
    Generates a concise natural language summary based on a user's semantic wildlife search.

    Args:
        user_query (str): Original semantic search input.

    Returns:
        str: Natural language summary about wildlife patterns, habitats, or regions.
    """
    prompt_text = (
        f"The user searched for '{user_query}'. "
        "Based on known wildlife sightings, what regions, behaviors, or species patterns are most associated? "
        "Provide a concise, insightful summary."
    )

    response = client.chat.completions.create(
        model=GPTMODEL,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant knowledgeable in wildlife habitats, species distribution, and sightings."},
            {"role": "user", "content": prompt_text}
        ]
    )

    return response.choices[0].message.content.strip()
