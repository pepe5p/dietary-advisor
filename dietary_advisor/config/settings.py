"""Centralized settings (pydantic-settings).

All knobs are read from environment / `.env` so the same code can be ablated
across LLM providers and reflection thresholds without code changes.
"""

from __future__ import annotations

from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_ai.models import Model
from pydantic_settings import BaseSettings, SettingsConfigDict

from dietary_advisor.config.llm import resolve_llm_model

# Shared constraint for the per-agent sampling temperature fields below.
Temperature = Annotated[float, Field(default=1.0, ge=0.0, le=2.0)]


class FoodDbUsage(str, Enum):
    """How a food DB participates in search (per source, `off`/`usda`).

    `disabled` also stops the pipeline from opening the reader at all; the
    channel modes select which of the two retrieval channels `search()` runs.
    """

    DISABLED = "disabled"
    ONLY_SEMANTIC = "only_semantic"
    ONLY_BM25 = "only_bm25"
    FULL = "full"


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
    llm_model: str = Field(...)
    # Sampling temperature for the nutrition/refiner/RAG agents. Provider defaults
    # tend to be conservative for tool-calling flows, which combined with fully
    # deterministic retrieval (BM25 + cosine similarity, no randomness) makes the
    # same "safe" ingredient choice recur across runs; raising this pushes back
    # without needing per-provider tuning in code.
    llm_temperature: Temperature
    # Separate knob for the meal-idea agent, so its dish-idea variety can be
    # tuned independently of the grounded nutrition/refiner/RAG agents above.
    meal_idea_llm_temperature: Temperature

    # --- Storage ---
    data_dir: Path = Field(default=Path(".data"))
    chroma_dir: Path = Field(default=Path(".data/rag/chroma"))
    # Chroma downloads its bundled MiniLM ONNX model (~80 MB) on first use.
    # Its default cache is under $HOME, which is ephemeral in the container;
    # keeping it under the bind-mounted .data means it's fetched only once.
    onnx_model_dir: Path = Field(default=Path(".data/rag/onnx_models"))

    # --- Open Food Facts (local product DB, built by `setup`) ---
    off_db: Path = Field(default=Path(".data/off/off_pl.duckdb"))
    # Semantic product search: fastembed model used to embed the per-product
    # document at build time and the query at runtime. BGE-M3 is multilingual
    # (Polish + English) and needs no query/passage prefixes.
    off_embedding_model: str = Field(default="BAAI/bge-m3")
    off_embedding_dim: int = Field(default=1024, gt=0)
    # fastembed downloads the ONNX model on first use; keep it under the
    # bind-mounted .data so it is fetched once (mirrors onnx_model_dir). Shared
    # by OFF and USDA, so it sits at the .data root rather than under off/.
    off_embedding_cache: Path = Field(default=Path(".data/fastembed"))
    # Which retrieval channel(s) the OFF reader uses (or `disabled` to skip it).
    off_usage: FoodDbUsage = Field(default=FoodDbUsage.FULL)

    # --- USDA FoodData Central (second local product DB, built by `setup`) ---
    # Foundation Foods (generic minimally-processed foods) + SR Legacy (the
    # frozen April 2018 Standard Reference). Complementary to OFF's branded
    # Polish products: these are the generic whole foods OFF largely lacks.
    # Same embedding model/dim/cache as OFF, so the query vectors are shared.
    usda_db: Path = Field(default=Path(".data/usda/usda.duckdb"))
    # Which retrieval channel(s) the USDA reader uses (or `disabled` to skip it).
    usda_usage: FoodDbUsage = Field(default=FoodDbUsage.FULL)

    # --- Reflection loop (critique-then-refine) ---
    reflection_max_loops: int = Field(default=3, ge=0, le=10)

    # --- RAG ---
    rag_top_k: int = Field(default=6, ge=1, le=50)
    rag_bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- LLM HTTP (Groq free tier often returns 429 with long Retry-After backoff) ---
    llm_request_timeout_s: float = Field(default=300.0, gt=0)
    llm_max_retries: int = Field(default=6, ge=0, le=20)

    def ensure_dirs(self) -> None:
        """Create the local data directories (idempotent)."""
        for p in (
            self.data_dir,
            self.chroma_dir,
            self.onnx_model_dir,
            self.off_embedding_cache,
        ):
            p.mkdir(parents=True, exist_ok=True)
        # Parent dirs for the duckdb files.
        for f in (self.off_db, self.usda_db):
            f.parent.mkdir(parents=True, exist_ok=True)

    def resolve_model(self, model_id: str) -> Model:
        """Resolve a pydantic-ai model identifier to a concrete `Model`."""
        return resolve_llm_model(
            model_id,
            groq_api_key=get_api_keys().groq_api_key,
            request_timeout_s=self.llm_request_timeout_s,
            max_retries=self.llm_max_retries,
        )

    @cached_property
    def resolved_llm_model(self) -> Model:
        """`llm_model` resolved to a concrete `Model`, cached for the process lifetime.

        Resolved lazily (on first access) rather than in `get_settings()` so that
        LLM-free commands (`setup`, `info`) don't fail on a missing API key.
        """
        return self.resolve_model(self.llm_model)


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s


@lru_cache(maxsize=1)
def get_api_keys() -> ApiKeys:
    return ApiKeys()
