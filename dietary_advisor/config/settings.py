"""Centralized settings (pydantic-settings).

All knobs are read from environment / `.env` so the same code can be ablated
across LLM providers and reflection thresholds without code changes.
"""

from __future__ import annotations

from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_ai.models import infer_model, Model
from pydantic_settings import BaseSettings, SettingsConfigDict

from dietary_advisor.config.llm import LlmSpec


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
    # "anthropic:claude-3-5-sonnet-latest", "ollama:llama3.1", optionally
    # suffixed with "#<effort>" (minimal|low|medium|high|xhigh) to request a
    # non-default reasoning effort - see `LlmSpec`.
    llm_model: str = Field(...)
    # The meal-idea brainstorm is a cheap, tool-less creative pass whose output is
    # never DB-verified, so it runs on its own small model independent of the
    # graded agents (and of the model the ablation harness sweeps).
    meal_idea_llm_model: str = Field(default="gemini-3.5-flash-lite")

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

    @property
    def llm_spec(self) -> LlmSpec:
        """`llm_model` parsed into a model id and an optional reasoning effort."""
        return LlmSpec.parse(self.llm_model)

    @cached_property
    def resolved_llm_model(self) -> Model:
        """`llm_model` resolved to a concrete `Model`, cached for the process lifetime.

        Resolved lazily (on first access) rather than in `get_settings()` so that
        LLM-free commands (`setup`) don't fail on a missing API key.
        """
        return infer_model(self.llm_spec.model)

    @cached_property
    def resolved_meal_idea_llm_model(self) -> Model:
        return infer_model(self.meal_idea_llm_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s
