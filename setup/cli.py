"""`setup` Typer CLI: idempotently provision local project data.

Runs an ordered list of steps - currently building the local Open Food
Facts product DB and ingesting the RAG clinical-guideline corpus - so
future provisioning (profile store init, Chroma warmup, ...) can be added
here without changing the invocation surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from dietary_advisor.config import get_settings
from setup.off_db import build_off_db
from setup.rag import build_rag_corpus

app = typer.Typer(help="Provision local project data (Open Food Facts DB, RAG corpus, and future setup steps).")

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def setup(
    force: bool = typer.Option(False, "--force", help="Rebuild local data even if already present."),
    raw_parquet: Path | None = typer.Option(
        None,
        "--raw-parquet",
        help="Filter this local OFF Parquet file instead of the downloaded cache.",
    ),
    redownload: bool = typer.Option(
        False,
        "--redownload",
        help="Re-fetch the OFF Parquet cache and RAG corpus PDFs even if already downloaded.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run all local setup steps, in order. Safe to re-run - each step skips if already done.

    The Open Food Facts export (~7.6 GB) is downloaded once to a local cache
    and filtered from there; it is not streamed on every run (that trips
    Hugging Face rate limits). `--redownload` also applies to the RAG corpus
    PDFs, re-fetching any that are already cached.
    """
    _configure_logging(verbose)
    settings = get_settings()

    log.info("Step 1/2: Open Food Facts product DB")
    build_off_db(settings, force=force, raw_parquet=raw_parquet, redownload=redownload)

    log.info("Step 2/2: RAG clinical-guideline corpus")
    stats = build_rag_corpus(settings, force=redownload)
    log.info(
        "Corpus: %d chunks indexed (failed: %s).",
        stats.chunks_indexed,
        stats.failed or "-",
    )

    log.info("Setup complete.")
