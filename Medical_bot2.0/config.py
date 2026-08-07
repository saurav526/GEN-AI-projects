"""
Centralized configuration for the Medical Chatbot.

All paths are resolved relative to this file, so the app works no matter
what directory it's launched from (a common bug when relative paths like
"vectorstore/db_faiss" are hardcoded in the app file).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"

# --- Model config --------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
GROQ_MODEL_NAME = os.environ.get(
    "GROQ_MODEL_NAME", "llama-3.3-70b-versatile"
)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "3"))

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""


def require_groq_api_key() -> str:
    """Fail fast with a clear message instead of a deep stack trace later."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file "
            "(see .env.example) or your environment/Streamlit secrets."
        )
    return GROQ_API_KEY