"""Centralized settings (pydantic-settings).

All knobs are read from environment / `.env` so the same code can be ablated
across LLM providers and reflection thresholds without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DA_",
        extra="ignore",
    )

    # --- LLM ---
    # pydantic-ai model identifier, e.g. "openai:gpt-4o-mini",
    # "anthropic:claude-3-5-sonnet-latest", "ollama:llama3.1".
    llm_model: str = Field(default="openai:gpt-4o-mini")
    # Optional secondary "judge" model used for the LLM-as-judge Faithfulness
    # metric. When unset, the judge falls back to llm_model.
    judge_model: str | None = None

    # --- Storage ---
    data_dir: Path = Field(default=Path(".data"))
    chroma_dir: Path = Field(default=Path(".data/chroma"))
    profile_db: Path = Field(default=Path(".data/profiles.sqlite"))
    usda_cache: Path = Field(default=Path(".data/usda_cache.sqlite"))
    corpus_dir: Path = Field(default=Path("dietary_advisor/knowledge/corpus"))

    # --- Reflection loop (Generate-Score-Refine) ---
    reflection_max_loops: int = Field(default=3, ge=0, le=10)
    entailment_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    factuality_threshold: float = Field(default=-1.0)

    # --- RAG ---
    rag_top_k: int = Field(default=6, ge=1, le=50)
    rag_bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rag_chunk_size: int = Field(default=900, ge=200, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=500)

    # --- LLM HTTP (Groq free tier often returns 429 with long Retry-After backoff) ---
    llm_request_timeout_s: float = Field(default=300.0, gt=0)
    llm_max_retries: int = Field(default=6, ge=0, le=20)

    # --- Misc ---
    request_timeout_s: float = Field(default=60.0, gt=0)

    def ensure_dirs(self) -> None:
        """Create the local data directories (idempotent)."""
        for p in (self.data_dir, self.chroma_dir, self.corpus_dir):
            p.mkdir(parents=True, exist_ok=True)
        # Parent dirs for sqlite files.
        for f in (self.profile_db, self.usda_cache):
            f.parent.mkdir(parents=True, exist_ok=True)


# A second settings group for raw third-party API keys. We read these as
# top-level env vars (no DA_ prefix) because that's how pydantic-ai / official
# SDKs expect them.
class ApiKeys(BaseSettings):
    """API keys for third-party services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    usda_api_key: str = Field(default="DEMO_KEY", alias="USDA_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


@lru_cache(maxsize=1)
def get_api_keys() -> ApiKeys:
    return ApiKeys()
