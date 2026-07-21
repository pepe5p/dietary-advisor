# Dietary Advisor

Neuro-symbolic agentic architecture for dietary recommendations (master's thesis).
An LLM is combined with deterministic symbolic components: a local Open Food Facts
lookup (always on - part of the baseline, since the agent cannot invent a food),
plus a `Fraction`-based nutrient totaller, hybrid RAG over clinical guidelines, and
a self-review reflection loop, each ablatable. Evaluated with a leave-one-out
ablation across three patient complexity levels (L1-L3).

## Requirements

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2 (the supported runtime)
- An LLM API key (OpenAI by default; Anthropic, Gemini and Groq also wired in via
  `pydantic-ai`) - or a locally-served OpenAI-compatible model (e.g. Ollama), no key needed
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

## Collect case runs (and evaluate later)

Case collection runs the full system plus one leave-one-out config per disabled
module against the scenarios in
[`evaluation/scenarios.py`](evaluation/scenarios.py) (profiles frozen in
[`evaluation/profiles/cases.py`](evaluation/profiles/cases.py), independent from
the CLI profiles above). The (model, variant, scenario) grid is hardcoded in
[`evaluation/case_runner/grid.py`](evaluation/case_runner/grid.py); re-running
skips any result file already present under `outputs/`.

```bash
just run-cases
```

Successful runs land as JSON under `outputs/` (visible on the host via the bind
mount). Scoring of those stored results (`just evaluate`) is pending a later
rework.

Full reproduction from a clean host:

```bash
just build && just dc setup && just dc run-cases
```

## Configuration

`docker-compose.yml` loads `.env` via `env_file`; all variables are read by
`dietary_advisor.config.Settings`, plus setup-only knobs (source URLs, download
caches, RAG chunking) in `setup.settings.SetupSettings` and the evaluation
judge model in `evaluation.settings.EvaluationSettings` - both extend the
shared `Settings`.

`DA_LLM_MODEL`/`DA_JUDGE_MODEL` are `pydantic-ai` model identifiers
(`openai:gpt-4o-mini`, `anthropic:claude-3-5-sonnet-latest`, `groq:llama-3.3-70b-versatile`,
...). Local models served through an OpenAI-compatible endpoint (e.g.
[Ollama](https://ollama.com/)) work the same way via `ollama:<tag>` plus
`OLLAMA_BASE_URL` - see the commented-out block in
[`.env.example`](.env.example). From inside the container, point
`OLLAMA_BASE_URL` at `http://host.docker.internal:11434/v1`, not `localhost`.

## Tests and lint

```bash
just test        # pytest (runs offline, no API key needed)
just lint_full   # ruff + fawltydeps + mypy
just all         # lint + tests
just all_ff      # lint + tests + fail fast 
```