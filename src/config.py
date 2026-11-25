"""Central configuration and path definitions for CourseRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HANDBOOK_DIR = DATA_DIR / "handbooks"
DEFAULT_PDF_PATH = HANDBOOK_DIR / "Python_DSA_Reference_Guide.pdf"
CACHE_DIR = DATA_DIR / "cache"
PARSE_CACHE_DIR = CACHE_DIR / "parsed_docs"
EMBED_CACHE_DIR = CACHE_DIR / "embeddings"
QDRANT_DIR = CACHE_DIR / "qdrant_store"

load_dotenv(PROJECT_ROOT / ".env")


def _clean_env(name: str, default: str = "") -> str:
    val = os.getenv(name, default).strip()
    placeholders = {
        "your-api-key-here",
        "replace-with-your-key",
        "replace-with-your-new-groq-api-key",
    }
    return "" if val.lower() in placeholders else val


@dataclass
class RAGSettings:
    """Runtime parameters controlling chunking, retrieval, and generation."""

    chunk_size: int = 850
    chunk_overlap: int = 140
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    collection_name: str = "courserag_chunks"

    retrieval_mode: str = "hybrid"  # "hybrid" | "dense" | "bm25"
    top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.60
    sparse_weight: float = 0.40

    llm_provider: str = field(default_factory=lambda: _clean_env("LLM_PROVIDER", "auto").lower())
    groq_api_key: str = field(default_factory=lambda: _clean_env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _clean_env("GROQ_MODEL", "openai/gpt-oss-120b"))

    gemini_api_key: str = field(default_factory=lambda: _clean_env("GEMINI_API_KEY") or _clean_env("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _clean_env("GEMINI_MODEL", "gemini-2.5-flash"))

    openai_api_key: str = field(default_factory=lambda: _clean_env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _clean_env("OPENAI_MODEL", "gpt-4o-mini"))

    temperature: float = 0.2
    max_tokens: int = 2200

    def resolve_active_provider(self) -> tuple[str, str, str]:
        """Return (provider, model_name, api_key) based on current configuration."""
        choice = (self.llm_provider or "auto").lower().strip()

        if choice == "local":
            return ("local", "extractive-mmr-v1", "")

        if choice == "groq":
            if self.groq_api_key:
                return ("groq", self.groq_model, self.groq_api_key)
            return ("local", "extractive-mmr-v1", "")

        if choice == "gemini":
            if self.gemini_api_key:
                return ("gemini", self.gemini_model, self.gemini_api_key)
            return ("local", "extractive-mmr-v1", "")

        if choice == "openai":
            if self.openai_api_key:
                return ("openai", self.openai_model, self.openai_api_key)
            return ("local", "extractive-mmr-v1", "")

        # Auto detection order: Groq -> Gemini -> OpenAI -> Local fallback
        if self.groq_api_key:
            return ("groq", self.groq_model, self.groq_api_key)
        if self.gemini_api_key:
            return ("gemini", self.gemini_model, self.gemini_api_key)
        if self.openai_api_key:
            return ("openai", self.openai_model, self.openai_api_key)

        return ("local", "extractive-mmr-v1", "")
