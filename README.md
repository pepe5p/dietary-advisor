# Dietary Advisor

Neuro-symbolic agentic architecture for dietary recommendations (master's thesis).
An LLM is combined with deterministic symbolic components: a local Open Food Facts
lookup (always on - part of the baseline, since the agent cannot invent a food),
plus a `Fraction`-based nutrient totaller, hybrid RAG over clinical guidelines, and
a self-review reflection loop, each ablatable. Evaluated with a leave-one-out
ablation across three patient complexity levels (L1-L3).

## Requirements

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2 (the supported runtime)
- An LLM API key (OpenAI by default; Anthropic and Gemini also wired in via `pydantic-ai`)
- Optional: [`just`](https://github.com/casey/just) on the host for the wrappers below;
  everything also works with bare `docker compose run --rm dietary_advisor ...`

## Setup

```bash
cp .env.example .env          # set at least one of OPENAI/ANTHROPIC/GEMINI_API_KEY
just build                    # docker compose build dietary_advisor
just dc bash                  # drop into the container shell (venv pre-loaded, at /code)
```

The rest of this README assumes you are inside the container shell (`just dc bash`).
The project is bind-mounted, so host edits are visible immediately with no rebuild.
For one-shot host commands without the shell, run any recipe via `just dc <recipe>`
(e.g. `just dc setup`, `just dc evaluate`).

Provision local data (Open Food Facts DB + clinical-guideline RAG corpus). Safe to
re-run; it skips anything already present:

```bash
just setup
```

Building the Open Food Facts DB also embeds every product (name, brands,
categories, ingredients, ...) with a local `fastembed` model
(`intfloat/multilingual-e5-small`) into a DuckDB VSS index, so product lookup is
hybrid BM25 + semantic search. The ~0.5 GB model is downloaded once on the first
`setup` run (cached under `.data/fastembed`); everything after that is
offline.

## Run the tool

`recommend` generates one plan. It takes exactly one of `--profile-id <id>`
(a built-in profile from
[`dietary_advisor/profiles.py`](dietary_advisor/profiles.py), `L1_01`…`L3_05`) or
`--profile '{...}'` (a full `UserProfile` JSON string).

```bash
just run recommend --profile-id L2_01 \
    --query "Plan a 1-day, ~1700 kcal vegetarian menu I can cook in 30 min." \
    --json-out out/plan.json

just run info                 # sanity-check resolved settings (model, data dir, .env)
just run recommend --help     # every command and option is self-documented
```

Results (macro targets, meal plan, shopping list, actual-vs-target macros, citation
count) print to the terminal; add `--json-out <path>` to also persist the full result.
Build a plan around what you have with `--available "salmon, spinach, lemon"`.

Each ablatable module can be toggled off to feel its effect:
`--no-totaller`, `--no-rag`, `--no-reflective-loop`.

## Run the evaluation

The ablation runs the full system plus one leave-one-out config per disabled module,
scored against ground-truth hard constraints frozen in
[`evaluation/profiles/cases.py`](evaluation/profiles/cases.py) (independent from the
CLI profiles above).

```bash
just evaluate --repeats 3
```

This writes three artefacts to `evaluation/reports/` (visible on the host via the bind
mount): `ablation.csv` (per-run rows), `ablation.md` (aggregated summary), and
`ablation.png` (CSR / SoftScore / MAE / MSE charts). With `--repeats 3`, 15 profiles ×
4 configs = 180 runs; expect 30-90 min with `gpt-4o-mini`. Add `--no-judge` to skip the
G-Eval soft-preference scoring.

Full reproduction from a clean host:

```bash
just build && just dc setup && just dc evaluate --repeats 3
```

## Configuration

`docker-compose.yml` loads `.env` via `env_file`; all variables are read by
`dietary_advisor.config.Settings`.

## Tests and lint

```bash
just test        # pytest (runs offline, no API key needed)
just lint_full   # ruff + fawltydeps + mypy
just all         # lint + tests
just all_ff      # lint + tests + fail fast 
```