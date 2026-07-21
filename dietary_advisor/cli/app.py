"""`dietary-advisor` Typer CLI: recommend / info."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.table import Table

from dietary_advisor.cli.rendering import console, render_result
from dietary_advisor.config import get_settings
from dietary_advisor.planning.pipeline import Pipeline, PipelineResult, VariantConfig
from dietary_advisor.profile import UserProfile
from dietary_advisor.profiles import get_profile

app = typer.Typer(help="Neuro-symbolic dietary advisor (master's thesis CLI).")

log = logging.getLogger(__name__)

# Set by the `--verbose`/`-v` top-level flag; gates the debug telemetry table
# (tool-call counts, LLM requests, token usage) printed by `render_result`.
_verbose = False


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Top-level options."""
    global _verbose
    _verbose = verbose
    _configure_logging(verbose)


# ---------- recommend ----------------------------------------------------------------------------


def _parse_ingredients(raw: str | None) -> list[str]:
    """Split a comma-separated ingredient string into a clean list."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_PROFILE_OPTION_HELP = (
    'Full UserProfile as a JSON string, e.g. \'{"user_id": "u1", "age": 30, "sex": "female", '
    '"height_cm": 165, "weight_kg": 60, "targets": {"energy_kcal": 2000, "protein_g": 90, '
    '"carbs_g": 250, "fat_g": 60}}\'. Mutually exclusive with --profile-id.'
)
_PROFILE_ID_OPTION_HELP = (
    "user_id of a built-in profile (see dietary_advisor/profiles.py). Mutually exclusive with --profile."
)


def _resolve_profile(profile: str | None, profile_id: str | None) -> UserProfile:
    """Resolve a `UserProfile` from the mutually exclusive --profile/--profile-id CLI inputs."""
    if (profile is None) == (profile_id is None):
        raise typer.BadParameter("Pass exactly one of --profile or --profile-id.")
    if profile_id is not None:
        try:
            return get_profile(profile_id)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from None
    assert profile is not None
    try:
        return UserProfile.model_validate_json(profile)
    except ValidationError as exc:
        raise typer.BadParameter(f"Invalid --profile JSON: {exc}") from None


def _result_payload(result: PipelineResult) -> dict[str, object]:
    """Serialise a `PipelineResult` to a JSON-friendly dict."""
    return {
        "variant": result.variant,
        "iterations": result.iterations,
        "plan": result.plan.model_dump(mode="json"),
        "targets": result.targets.model_dump(mode="json"),
        "citations": [c.model_dump(mode="json") for c in result.citations],
        "shopping_list": result.shopping_list.model_dump(mode="json"),
        "telemetry": result.telemetry.as_dict(),
    }


@app.command()
def recommend(
    profile: str | None = typer.Option(None, "--profile", help=_PROFILE_OPTION_HELP),
    profile_id: str | None = typer.Option(None, "--profile-id", help=_PROFILE_ID_OPTION_HELP),
    query: str = typer.Option(
        "Plan one balanced day of meals.",
        "--query",
        "-q",
        help="The user-facing prompt forwarded to the LLM.",
    ),
    no_totaller: bool = typer.Option(False, "--no-totaller", help="Disable the deterministic totaller tool."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable clinical-guideline retrieval (RAG)."),
    no_reflective_loop: bool = typer.Option(
        False,
        "--no-reflective-loop",
        help="Disable the critique-then-refine reflection loop.",
    ),
    available: str | None = typer.Option(
        None,
        "--available",
        "-a",
        help="Comma-separated ingredients to build the plan around (e.g. fridge contents).",
    ),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the full result to JSON."),
) -> None:
    """Generate a single recommendation for a profile (full system by default)."""
    resolved_profile = _resolve_profile(profile, profile_id)

    variant = VariantConfig(
        totaller_enabled=not no_totaller,
        rag_enabled=not no_rag,
        reflection_enabled=not no_reflective_loop,
    )
    ingredients = _parse_ingredients(available)
    with Pipeline(variant) as pipeline:
        result = asyncio.run(pipeline.run(resolved_profile, query, available_ingredients=ingredients))

    render_result(result, verbose=_verbose)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(_result_payload(result), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {json_out}")


# ---------- info -------------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Print configuration sanity-check info."""
    settings = get_settings()
    tbl = Table(title="dietary-advisor settings")
    tbl.add_column("key")
    tbl.add_column("value")
    for k, v in settings.model_dump().items():
        tbl.add_row(k, str(v))
    console.print(tbl)


if __name__ == "__main__":  # pragma: no cover
    app()
