# Dietary Advisor

Neuro-symbolic agentic architecture for dietary recommendations - master's thesis
implementation. The system integrates Large Language Models with deterministic
symbolic components (USDA nutrient lookup, MILP-based diet optimisation, rule-based
validation, hybrid RAG over clinical guidelines) and supports a full ablation study
across five system variants (V0-V4) and three patient complexity levels (L1-L3).

## What's inside

| Module | Purpose |
| --- | --- |
| `dietary_advisor/schemas/` | Pydantic DSL: `UserProfile`, `MealPlan`, `HardConstraint`, `ValidationReport`, `Citation`, ... |
| `dietary_advisor/tools/` | Deterministic computation: TDEE, totaller (precise nutrient summation via `Fraction`), USDA FDC client, MILP optimiser (PuLP) |
| `dietary_advisor/profile_manager/` | SQLite-backed CRUD for profiles and constraint derivation |
| `dietary_advisor/knowledge/` | Hybrid (Chroma + BM25) RAG over clinical guideline PDFs |
| `dietary_advisor/agents/` | `pydantic-ai` agents: nutrition, RAG, profile, refiner |
| `dietary_advisor/validation/` | Rule-based validator + Generate-Score-Refine reflection loop |
| `dietary_advisor/pipeline.py` | Top-level orchestrator with `VariantConfig` feature flags |
| `dietary_advisor/cli.py` | `dietary-advisor` Typer CLI |
| `evaluation/` | Test profiles, scenarios, metrics (HSR/SSR/CSR, MAE/MSE, Faithfulness), batch runner, report generator |
| `tests/` | Unit + smoke tests (51 tests, ~69% coverage) |

## Ablation variants

| Variant | Totaller | Profile constraints | RAG | Reflection loop |
| --- | :-: | :-: | :-: | :-: |
| **V0** baseline LLM | x | x | x | x |
| **V1** + Totaller | OK | x | x | x |
| **V2** + Profile constraints | OK | OK | x | x |
| **V3** + RAG | OK | OK | OK | x |
| **V4** + Reflection (full) | OK | OK | OK | OK |

## Requirements

- Python 3.13 (managed by [`uv`](https://docs.astral.sh/uv/))
- macOS / Linux (no Docker required)
- An LLM API key (OpenAI by default; Anthropic and Gemini are also wired in via `pydantic-ai`)
- Optional: a USDA FoodData Central API key (free at <https://fdc.nal.usda.gov/api-key-signup>) - without it the pipeline runs in offline mode and skips USDA lookups

## Installation

```bash
# Clone, then from the repo root:
uv sync
```

This creates `.venv/` and installs all runtime + dev dependencies.

## Configuration

```bash
cp .env.example .env
# Edit .env and set at minimum OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY).
```

Key environment variables (all are read by `dietary_advisor.config.Settings`):

| Variable | Default | Notes |
| --- | --- | --- |
| `DA_LLM_MODEL` | `openai:gpt-4o-mini` | Any `pydantic-ai` model id |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | - | At least one is required |
| `USDA_API_KEY` | - | Optional; enables USDA FDC lookups |
| `DA_DATA_DIR` | `.data/` | Root for ChromaDB / SQLite / cache |
| `DA_REFLECTION_MAX_LOOPS` | `3` | Max iterations of the Generate-Score-Refine loop |
| `DA_RAG_TOP_K` | `5` | Top-k retrieved chunks |
| `DA_RAG_BM25_WEIGHT` | `0.3` | Weight in the dense+sparse fusion |

## Quick start

```bash
# 1. Load the 15 bundled test profiles into the local SQLite store
uv run dietary-advisor profile import-dir evaluation/profiles

# 2. Build the RAG corpus (downloads PDFs where possible, falls back to seed
#    excerpts shipped under dietary_advisor/knowledge/seeds/)
uv run dietary-advisor ingest-corpus

# 3. Run a single recommendation, e.g. V4 (full system) for a Type-2 diabetic
uv run dietary-advisor recommend L3_t2dm_male --variant V4 \
    --query "Plan a 1-day, 1800 kcal menu suitable for me." \
    --json-out out/L3_t2dm_male_V4.json

# 4. Run the full ablation grid (V0..V4 x L1..L3 x all profiles, 1 repeat)
uv run dietary-advisor evaluate \
    --variants V0,V1,V2,V3,V4 \
    --levels 1,2,3 \
    --output evaluation/reports/ablation.csv
```

The `evaluate` command produces three artefacts in `evaluation/reports/`:

- `ablation.csv` - raw per-run rows (variant x profile x repeat)
- `ablation.md` - aggregated Markdown summary table (mean/std per variant, per level)
- `ablation.png` - matplotlib bar charts of HSR, CSR, MAE-kcal, Faithfulness

## Reproduce the full ablation study (one command)

```bash
uv run dietary-advisor profile import-dir evaluation/profiles \
  && uv run dietary-advisor ingest-corpus \
  && uv run dietary-advisor evaluate --variants V0,V1,V2,V3,V4 --levels 1,2,3 --repeats 3
```

Reports land in `evaluation/reports/ablation.{csv,md,png}` and can be pasted
directly into the thesis. With `--repeats 3` and 15 profiles x 5 variants the
grid contains 225 runs; expect 30-90 minutes wall clock with `gpt-4o-mini`.

## Profile management

```bash
uv run dietary-advisor profile list                  # list all profiles
uv run dietary-advisor profile add path/to/p.json    # add one profile
uv run dietary-advisor profile show L1_active_male   # dump JSON
uv run dietary-advisor profile delete L1_active_male
```

Profiles live in `evaluation/profiles/` with file-name conventions:

- `L1_*` - healthy adults (no clinical conditions, no allergens)
- `L2_*` - dietary preferences and/or allergens (vegan, vegetarian, peanut allergy, ...)
- `L3_*` - clinical conditions (T2DM, hypertension, CKD, hyperlipidaemia, ...)

## Clinical knowledge corpus

`dietary_advisor/knowledge/sources.py` lists the source documents the ingester
attempts to download:

- WHO healthy-diet fact sheet
- WHO 2023 global report on sodium intake reduction
- USDA Dietary Guidelines for Americans 2020-2025
- ADA Nutrition Therapy for Adults With Diabetes (2019)
- NICE NG28 (T2 diabetes), NG136 (hypertension)
- EFSA Dietary Reference Values - summary

If a URL is unreachable, the ingester logs the failure and falls back to the
plain-text excerpt under `dietary_advisor/knowledge/seeds/` so the system always
has at least *some* grounding material. PDFs themselves are gitignored; only the
seed excerpts are tracked.

## Tests, lint, coverage

```bash
just test         # uv run pytest
just lint_full    # ruff + fawltydeps + ...
just all          # lint + tests
```

The suite contains 51 tests covering schema validation, TDEE math, totaller
fractional precision, MILP optimality and feasibility, profile/store CRUD,
validator rules, ablation metrics, the Typer CLI, and an offline pipeline
smoke test that mocks every LLM call via `pydantic_ai.models.test.TestModel`
- so the full test suite runs without any API key. Coverage threshold is 60%.

## Architecture notes

- **No Docker.** Everything runs locally via `uv` for portability and faster
  iteration during the thesis.
- **LLM abstraction** is delegated to `pydantic-ai`, so swapping `openai:gpt-4o-mini`
  for `anthropic:claude-3-5-sonnet` or `google-gla:gemini-2.0-flash` is a
  one-line `.env` change.
- **Determinism where it matters.** Nutrient totalling uses
  `fractions.Fraction` to avoid floating-point drift; constraint validation is
  rule-based code (not an LLM judge); the MILP optimiser uses PuLP's CBC backend.
- **Reflection loop.** When V4 detects a hard-constraint violation, it sends
  the structured violation report back to a refiner agent (configurable
  `DA_REFLECTION_MAX_LOOPS`, default 3).
- **Variant control** is a single `VariantConfig` dataclass of feature flags
  consumed by `Pipeline.run` - keeping the ablation reproducible and the call
  graph easy to audit.
