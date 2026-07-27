"""`dietary-advisor` Typer CLI: generate a recommendation for a built-in profile."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer

from dietary_advisor.cli.rendering import console, render_result
from dietary_advisor.planning.pipeline import Pipeline, PipelineResult, VariantConfig
from dietary_advisor.profiles import get_profile, PROFILES

app = typer.Typer()

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


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


@app.command(
    help=(
        "Neuro-symbolic dietary advisor (master's thesis CLI). "
        "Generate a single recommendation for a profile (full system by default)."
    ),
    no_args_is_help=True,
)
def main(
    profile_id: str = typer.Argument(..., help="user_id of a built-in profile (see dietary_advisor/profiles.py)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    query: str = typer.Option(
        "",
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
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the full result to JSON."),
) -> None:
    _configure_logging(verbose)

    try:
        resolved_profile = get_profile(profile_id)
    except KeyError:
        raise typer.BadParameter(
            f"Unknown profile id. Available: {', '.join(sorted(PROFILES))}",
            param_hint="PROFILE_ID",
        ) from None

    variant = VariantConfig(
        totaller_enabled=not no_totaller,
        rag_enabled=not no_rag,
        reflection_enabled=not no_reflective_loop,
    )
    with Pipeline(variant) as pipeline:
        result = asyncio.run(pipeline.run(resolved_profile, query))

    render_result(result, verbose=verbose)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(_result_payload(result), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {json_out}")


if __name__ == "__main__":  # pragma: no cover
    app()
