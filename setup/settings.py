"""Setup-only configuration: source URLs, download caches, RAG chunking.

Extends the shared `Settings` so `setup` code gets both the runtime knobs
(e.g. `off_db`, `off_embedding_model`) and the build-only ones below from a
single object, matching how the `setup` build functions already take one
`settings` argument.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

from dietary_advisor.config.settings import Settings


class SetupSettings(Settings):
    """Configuration for the one-off `setup` provisioning steps."""

    # --- RAG corpus ingest ---
    corpus_dir: Path = Field(default=Path(".data/rag/corpus"))
    rag_chunk_size: int = Field(default=900, ge=200, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=500)
    # Timeout for downloading corpus source PDFs (distinct from `llm_request_timeout_s`).
    request_timeout_s: float = Field(default=60.0, gt=0)

    # --- Open Food Facts export ---
    # Local cache of the full OFF Parquet export (~7.6 GB). Downloaded once and
    # filtered locally; streaming it remotely trips Hugging Face rate limits.
    off_raw_parquet: Path = Field(default=Path(".data/off/food.parquet"))
    off_source_url: str = Field(
        default="https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
    )

    # --- USDA FoodData Central export ---
    # Local cache dir for the downloaded FDC CSV zips (extracted in place).
    usda_raw_dir: Path = Field(default=Path(".data/usda/raw"))
    # FDC ships fixed-vintage zips; SR Legacy is final (April 2018). URLs are
    # overridable so a newer Foundation vintage can be swapped in without code
    # changes (the build verifies the archive still has the CSVs it needs).
    usda_foundation_source_url: str = Field(
        default="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2024-10-31.zip"
    )
    usda_sr_legacy_source_url: str = Field(
        default="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"
    )

    def ensure_dirs(self) -> None:
        super().ensure_dirs()
        for p in (self.corpus_dir, self.usda_raw_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.off_raw_parquet.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_setup_settings() -> SetupSettings:
    s = SetupSettings()  # type: ignore[call-arg]
    s.ensure_dirs()
    return s
