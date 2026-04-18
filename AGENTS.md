# AGENTS.md

Operational guide for AI coding agents (Cursor, Claude Code, Codex, ...) working
in this repository. Humans should read `README.md` first; this file captures the
implicit conventions that aren't obvious from the code.

## Project at a glance

- **What:** Neuro-symbolic agentic dietary recommender (master's thesis). LLMs
  (`pydantic-ai`) wrapped around deterministic tools: USDA FDC lookup, MILP
  optimiser (PuLP/CBC), `Fraction`-based nutrient totaller, rule-based
  validator, hybrid (Chroma + BM25) RAG, Generate-Score-Refine reflection loop.
- **Why it exists:** an ablation study across 5 system variants (V0..V4) and
  3 patient complexity levels (L1..L3). Reproducibility of the ablation is the
  top priority - do not introduce non-determinism in the symbolic components.
- **Top-level orchestrator:** `dietary_advisor/pipeline.py` (`Pipeline.run`,
  driven by a `VariantConfig` dataclass of feature flags).
- **CLI entry point:** `dietary_advisor/cli.py` (Typer; exposed as
  `dietary-advisor` and `python -m dietary_advisor`).

## Module map

| Path | Purpose |
| --- | --- |
| `dietary_advisor/schemas/` | Pydantic DSL (`UserProfile`, `MealPlan`, `HardConstraint`, `ValidationReport`, `Citation`). |
| `dietary_advisor/tools/` | Deterministic computation: TDEE, totaller, USDA client, MILP. |
| `dietary_advisor/profile_manager/` | SQLite-backed profile CRUD + constraint derivation. |
| `dietary_advisor/knowledge/` | Hybrid RAG over clinical guideline PDFs (with seed-text fallback). |
| `dietary_advisor/agents/` | `pydantic-ai` agents: nutrition, RAG, profile, refiner. |
| `dietary_advisor/validation/` | Rule-based validator + reflection loop. |
| `evaluation/` | Test profiles, scenarios, metrics (HSR/SSR/CSR, MAE/MSE, Faithfulness), batch runner, report generator. |
| `tests/` | Unit + smoke tests. Pipeline smoke test mocks LLMs via `pydantic_ai.models.test.TestModel` - no API key required. |

## Runtime model - read this before running anything

**Docker is the supported runtime.** The host's `.venv` is shadowed inside the
container by an anonymous volume, so do NOT run `uv sync` / `pytest` / etc.
on the host expecting the container's environment. Use `just dc <recipe>` or
`docker compose run --rm dietary_advisor ...`.

Common recipes (all wrap `docker compose run --rm`):

```bash
just build               # build the image
just dc test             # pytest in the container
just dc lint_full        # ruff + fawltydeps + mypy in the container
just dc all              # lint_full + test
just dc bash             # drop into the container shell
just cli <args>          # run the CLI inside the container
```

If you must work on the host (e.g. quick iteration in editor): `uv sync` first,
then `uv run pytest`. CI and the canonical "did it pass?" answer is the
container.

## Coding conventions

- **Python 3.13**, fully type-annotated. `mypy` runs with
  `disallow_untyped_defs = true`. Add types to every new function/method.
- **Ruff** is the formatter and linter (line length 120). Active rule sets in
  `pyproject.toml` include `B`, `S`, `SIM`, `N`, `I`, `T20` (no stray prints),
  `G`/`LOG` (logging-format), `PT` (pytest style), etc. Run `just dc lint_fix`
  before proposing a diff.
- **Imports:** isort via Ruff, `dietary_advisor` is first-party.
- **No `print` in library code** - use the `logging` module. `T20` will flag it.
  CLI output via `rich` / `typer.echo` is fine inside `cli.py`.
- **Determinism in symbolic code:** use `fractions.Fraction` for nutrient
  arithmetic (see `tools/totaller.py`). Don't reintroduce floats there.
- **Pydantic v2** for all data structures crossing module boundaries. New
  schemas go under `dietary_advisor/schemas/` and should be re-exported from
  the package's `__init__`.
- **`pydantic-ai` agents:** keep prompts in code (not external files) so the
  ablation is self-contained. New agents follow the pattern in
  `dietary_advisor/agents/` and should accept an injected model for testability.
- **Variant flags:** any new behaviour that should be ablatable must be
  gated behind a field on `VariantConfig`, not a global toggle.

## Testing

- `pytest` is configured in `pyproject.toml`: `testpaths = "tests"`,
  `--cov-fail-under=60`, `asyncio_mode = "auto"`.
- LLM calls in tests must be mocked with
  `pydantic_ai.models.test.TestModel` (see `tests/test_pipeline_smoke.py` and
  `tests/conftest.py`). Tests must pass with no API keys set.
- Add a unit test for any new tool/validator/metric; add a smoke test branch
  for any new `VariantConfig` flag.

## Things NOT to do

- Don't commit downloaded corpus PDFs or anything under `.data/` - both are
  gitignored. Only the curated text excerpts under
  `dietary_advisor/knowledge/seeds/` are tracked.
- Don't hard-code an LLM provider. Read `DA_LLM_MODEL` from
  `dietary_advisor.config.Settings` so swapping `openai:gpt-4o-mini` for
  Anthropic/Gemini stays a one-line `.env` change.
- Don't bypass the validator with an LLM-as-judge. Hard-constraint checking is
  deliberately rule-based.
- Don't introduce non-determinism (random seeds, time-based IDs, dict ordering
  reliance) into the totaller, MILP, or validator.
- Don't add new top-level dependencies without also updating `pyproject.toml`
  and the `fawltydeps` allow-lists if needed; `just dc deps` must stay green.

## When in doubt

- Read `dietary_advisor/pipeline.py` first - it's the single call graph that
  ties everything together.
- For ablation/evaluation questions, see `evaluation/runner.py` and
  `evaluation/metrics.py`.
- Reference docs and source URLs for clinical guidelines live in
  `dietary_advisor/knowledge/sources.py`.
