"""`setup` Typer CLI: idempotently provision local project data.

Runs an ordered list of steps - currently building the local Open Food
Facts product DB and ingesting the RAG clinical-guideline corpus - so
future provisioning (profile store init, Chroma warmup, ...) can be added
here without changing the invocation surface.
"""

from __future__ import annotations

import logging

import typer

from dietary_advisor.config import get_settings, Settings
from setup.duckdb_creation import build_off_db
from setup.rag import build_rag_corpus
from setup.usda_duckdb_creation import build_usda_db

app = typer.Typer(
    help="Provision local project data (Open Food Facts DB, USDA DB, RAG corpus, and future setup steps)."
)

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _step_off(settings: Settings) -> None:
    log.info("Open Food Facts product DB")
    build_off_db(settings)


def _step_usda(settings: Settings) -> None:
    log.info("USDA FoodData Central product DB")
    build_usda_db(settings)


def _step_rag(settings: Settings) -> None:
    log.info("RAG clinical-guideline corpus")
    stats = build_rag_corpus(settings)
    log.info(
        "Corpus: %d chunks indexed (failed: %s).",
        stats.chunks_indexed,
        stats.failed or "-",
    )


# Ordered so `setup` runs steps in dependency order; keys are the CLI step names.
_STEPS = {
    "off": _step_off,
    "usda": _step_usda,
    "rag": _step_rag,
}

_VERBOSE = typer.Option(False, "--verbose", "-v")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, verbose: bool = _VERBOSE) -> None:
    """Run all local setup steps, in order. Safe to re-run - each step skips if already done.

    Invoke a subcommand (e.g. ``off``, ``usda``, ``rag``) to run only that step.

    The Open Food Facts export (~7.6 GB) is downloaded once to a local cache
    and filtered from there; it is not streamed on every run (that trips
    Hugging Face rate limits).
    """
    if ctx.invoked_subcommand is not None:
        return

    _configure_logging(verbose)
    settings = get_settings()

    total = len(_STEPS)
    for i, (name, step) in enumerate(_STEPS.items(), start=1):
        log.info("Step %d/%d: %s", i, total, name)
        step(settings)

    log.info("Setup complete.")


def _run_step(name: str, verbose: bool) -> None:
    _configure_logging(verbose)
    _STEPS[name](get_settings())
    log.info("Step %r complete.", name)


@app.command()
def off(verbose: bool = _VERBOSE) -> None:
    """Build only the Open Food Facts product DB."""
    _run_step("off", verbose)


@app.command()
def usda(verbose: bool = _VERBOSE) -> None:
    """Build only the USDA FoodData Central product DB."""
    _run_step("usda", verbose)


@app.command()
def rag(verbose: bool = _VERBOSE) -> None:
    """Ingest only the RAG clinical-guideline corpus."""
    _run_step("rag", verbose)
