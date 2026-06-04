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

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2
- An LLM API key (OpenAI by default; Anthropic and Gemini are also wired in via `pydantic-ai`)
- Optional: a USDA FoodData Central API key (free at <https://fdc.nal.usda.gov/api-key-signup>) - without it the pipeline runs in offline mode and skips USDA lookups
- Optional: [`just`](https://github.com/casey/just) on the host for the convenience wrappers below; everything also works with bare `docker compose` commands

## Configuration

```bash
cp .env.example .env
# Edit .env and set at minimum OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY).
```

`docker-compose.yml` reads `.env` via `env_file`, so any variable defined there is
available inside the container without further wiring.

Key environment variables (all are read by `dietary_advisor.config.Settings`):

| Variable | Default | Notes |
| --- | --- | --- |
| `DA_LLM_MODEL` | `openai:gpt-4o-mini` | Any `pydantic-ai` model id |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | - | At least one is required |
| `USDA_API_KEY` | - | Optional; enables USDA FDC lookups |
| `DA_DATA_DIR` | `/code/.data` (in image) | Bind-mounted to host `./.data` |
| `DA_REFLECTION_MAX_LOOPS` | `3` | Max iterations of the Generate-Score-Refine loop |
| `DA_RAG_TOP_K` | `5` | Top-k retrieved chunks |
| `DA_RAG_BM25_WEIGHT` | `0.3` | Weight in the dense+sparse fusion |

## Build the image

```bash
docker compose build dietary_advisor
# or, with just:
just build
```

The image is single-purpose: it ships dev + runtime dependencies and is meant to be
invoked as a one-shot CLI runner via `docker compose run --rm dietary_advisor ...`.

## Quick start

The supported workflow is to drop into the container shell once and then drive
the CLI directly as `dietary-advisor <cmd>`. Inside the image, the CLI is
exposed on `PATH` so the in-container UX is the same as a locally installed
tool:

```bash
just dc bash
# you are now inside the container, at /code, with the venv pre-loaded:

# 1. Load the 15 bundled test profiles into the local SQLite store
dietary-advisor profile import-dir evaluation/profiles

# 2. Build the RAG corpus (downloads PDFs where possible, falls back to seed
#    excerpts shipped under dietary_advisor/knowledge/seeds/)
dietary-advisor ingest-corpus

# 3. Run a single recommendation, e.g. V4 (full system) for a Type-2 diabetic
#    Results print to the terminal; add `--json-out out/...json` to also
#    persist the full structured result.
dietary-advisor recommend L3_t2dm_male --variant V4 \
    --query "Plan a 1-day, 1800 kcal menu suitable for me."

# 4. Run the full ablation grid (V0..V4 x L1..L3 x all profiles, 1 repeat)
dietary-advisor evaluate \
    --variants V0,V1,V2,V3,V4 \
    --levels 1,2,3 \
    --output evaluation/reports/ablation.csv
```

### One-shot from the host

For scripting or single commands you don't need the interactive shell. Each
of the commands above also runs as a one-shot from the host via the `just cli`
wrapper, which spins up a fresh container per invocation:

```bash
just cli profile import-dir evaluation/profiles
just cli recommend L3_t2dm_male --variant V4 \
    --query "Plan a 1-day, 1800 kcal menu suitable for me."
```

The bare-`docker compose` equivalent (no `just` on the host) is identical in
shape:

```bash
docker compose run --rm dietary_advisor \
    dietary-advisor profile import-dir evaluation/profiles
```

The `evaluate` command produces three artefacts in `evaluation/reports/` (visible
on the host thanks to the `./:/code/` bind mount):

- `ablation.csv` - raw per-run rows (variant x profile x repeat)
- `ablation.md` - aggregated Markdown summary table (mean/std per variant, per level)
- `ablation.png` - matplotlib bar charts of HSR, CSR, MAE-kcal, Faithfulness

## Use the CLI interactively (as a human, not a benchmark)

The commands in the quick start target the ablation study (bundled profiles,
batch grid). If you just want to ask the system for a plan for *yourself*, the
loop is:

1. **Drop into the container.** All subsequent commands assume you are inside
   the container shell:

   ```bash
   just dc bash
   ```

   You will land in `/code` with the project bind-mounted from the host, the
   pre-built venv on `PATH`, and `dietary-advisor` available as a real
   command. Edits made on the host (e.g. to `my_profile.json` below) are
   visible immediately - no rebuild needed.

2. **Discover commands.** Every command and option is self-documented:

   ```bash
   dietary-advisor --help
   dietary-advisor recommend --help
   dietary-advisor profile --help
   dietary-advisor info            # sanity-check resolved settings (.env, model, data dir)
   ```

3. **Describe yourself in a JSON profile.** Copy any file under
   `evaluation/profiles/` as a template and edit the fields - all of them are
   defined by `UserProfile` in `dietary_advisor/schemas/profile.py`. A minimal
   example (`my_profile.json` in the repo root, so the bind-mount picks it up):

   ```json
   {
     "user_id": "me",
     "name": "Your Name",
     "age": 32,
     "sex": "female",
     "height_cm": 168,
     "weight_kg": 64,
     "activity_factor": 1.4,
     "allergens": ["peanut"],
     "conditions": ["none"],
     "diet_pattern": "vegetarian",
     "disliked_foods": ["olives"],
     "preferred_foods": ["lentils", "yogurt"],
     "goal": {"kind": "lose", "weekly_rate_kg": 0.25}
   }
   ```

   Allowed values for `sex`, `activity_factor`, `allergens`, `conditions`,
   `diet_pattern`, and `goal.kind` are enumerated in
   `dietary_advisor/schemas/profile.py` (the CLI tells you if a value is off).

4. **Import the profile and ask for a plan.** `recommend` looks up the
   profile by `user_id`, so the import step is a one-off:

   ```bash
   dietary-advisor profile add my_profile.json
   dietary-advisor profile show me            # verify what got stored

   dietary-advisor recommend me \
       --query "Plan a 1-day, ~1700 kcal vegetarian menu I can cook in 30 min." \
       --variant V4
   ```

   Everything is printed to the terminal: macro targets, the meal plan, the
   validator verdict (HSR plus any hard-constraint violations), and a
   citation count. If you want the full structured result on disk for further
   inspection, add `--json-out out/me_V4.json` - it dumps plan, validation
   report, retrieved citations, and derived constraints as JSON.

5. **Iterate.** Re-run `recommend` with a different `--query` to ask
   follow-ups ("make breakfast lower-carb", "swap the lunch protein"), or
   with a lower `--variant` (e.g. `V0`, `V1`) to feel the effect of each
   symbolic component. Re-run `dietary-advisor profile add my_profile.json`
   after editing the file - the importer upserts on `user_id`. Use
   `dietary-advisor profile delete me` to start over.

For a richer pool of citations, run `dietary-advisor ingest-corpus` once
before step 4 - it downloads the clinical-guideline PDFs where reachable and
falls back to the bundled seed excerpts otherwise.

> Prefer one-shots from the host? Every `dietary-advisor <args>` here works
> as `just cli <args>` from the host (or `docker compose run --rm
> dietary_advisor dietary-advisor <args>` without `just`). The interactive
> shell just avoids paying the container-startup cost on every command.

## Reproduce the full ablation study (one command)

```bash
just build \
  && just cli ingest-corpus \
  && just cli evaluate --variants V0,V1,V2,V3,V4 --levels 1,2,3 --repeats 3
```

Reports land in `evaluation/reports/ablation.{csv,md,png}` on the host and can be
pasted directly into the thesis. With `--repeats 3` and 15 profiles x 5 variants
the grid contains 225 runs; expect 30-90 minutes wall clock with `gpt-4o-mini`.

## Profile management

```bash
just cli profile list                     # list all profiles
just cli profile add path/to/p.json       # add one profile
just cli profile show L1_active_male      # dump JSON
just cli profile delete L1_active_male
```

Ablation profiles are frozen in [`evaluation/profiles/cases.py`](evaluation/profiles/cases.py)
as `EvalProfile` entries (`L1_*` … `L3_*` case ids encode complexity level).

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
seed excerpts are tracked. Both PDFs and the ChromaDB index land under
`./.data/` and `dietary_advisor/knowledge/corpus/` on the host.

## Tests, lint, coverage

```bash
just dc test       # pytest in the container
just dc lint_full  # ruff + fawltydeps + mypy in the container
just dc all        # lint + tests in the container
```

Bare equivalents:

```bash
docker compose run --rm dietary_advisor just test
docker compose run --rm dietary_advisor just lint_full
docker compose run --rm dietary_advisor just all
```

The suite contains 51 tests covering schema validation, TDEE math, totaller
fractional precision, MILP optimality and feasibility, profile/store CRUD,
validator rules, ablation metrics, the Typer CLI, and an offline pipeline
smoke test that mocks every LLM call via `pydantic_ai.models.test.TestModel`
- so the full test suite runs without any API key. Coverage threshold is 60%.

## Drop into a shell

```bash
just dc bash
# or:
docker compose run --rm dietary_advisor bash
```

Inside the container both the CLI and the dev tooling are on `PATH`, so you
can iterate without leaving the shell:

```bash
dietary-advisor recommend me --variant V4   # the CLI
just test                                   # pytest
just lint_full                              # ruff + fawltydeps + mypy
```

## Architecture notes

- **Docker is the supported runtime.** A single image (built by
  `docker compose build`) carries both the CLI and the full dev/test toolchain.
  `.venv` is built once during the image build and preserved across runs via an
  anonymous volume that masks the host's macOS `.venv`.
- **State on the host.** `./` is bind-mounted into `/code/`, so `.data/` (Chroma
  index, profile SQLite, USDA cache), downloaded corpus PDFs, and
  `evaluation/reports/` artefacts all persist on the host and are easy to
  inspect/version.
- **LLM abstraction** is delegated to `pydantic-ai`, so swapping
  `openai:gpt-4o-mini` for `anthropic:claude-3-5-sonnet` or
  `google-gla:gemini-2.0-flash` is a one-line `.env` change.
- **Determinism where it matters.** Nutrient totalling uses
  `fractions.Fraction` to avoid floating-point drift; constraint validation is
  rule-based code (not an LLM judge); the MILP optimiser uses PuLP's CBC backend.
- **Reflection loop.** When V4 detects a hard-constraint violation, it sends
  the structured violation report back to a refiner agent (configurable
  `DA_REFLECTION_MAX_LOOPS`, default 3).
- **Variant control** is a single `VariantConfig` dataclass of feature flags
  consumed by `Pipeline.run` - keeping the ablation reproducible and the call
  graph easy to audit.
