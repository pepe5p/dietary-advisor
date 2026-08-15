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

It takes exactly one of profile id 
(a built-in profile from [`dietary_advisor/profiles.py`](dietary_advisor/profiles.py))

```bash
just run vegetarian-allergic \
    --query "Plan a 1-day, ~1700 kcal vegetarian menu I can cook in 30 min." \
    --json-out out/plan.json

just run --help               # every command and option is self-documented
```

Results (macro targets, meal plan, shopping list, actual-vs-target macros, citation
count) print to the terminal; add `--json-out <path>` to also persist the full result.

Each ablatable module can be toggled off to feel its effect:
`--no-totaller`, `--no-reflective-loop`. Add `--rag` to enable clinical-guideline retrieval.

## Collect case runs (and evaluate later)

Case collection runs the full system plus one leave-one-out config per disabled
module against the scenarios in
[`evaluation/scenarios.py`](evaluation/scenarios.py) (profiles from
[`dietary_advisor/profiles.py`](dietary_advisor/profiles.py), the same list the
CLI uses). The (model, variant, scenario) grid is hardcoded in
[`evaluation/case_runner/grid.py`](evaluation/case_runner/grid.py); re-running
skips any result file already present under the configured output directory
(project-top-level `outputs/runs/` by default, overridable via `DA_OUTPUT_DIR`).

```bash
just run_cases
```

Successful runs land as JSON under that directory (visible on the host via the
bind mount). Scoring of those stored results (`just evaluate`) is pending a
later rework.

Full reproduction from a clean host:

```bash
just build && just dc setup && just dc run_cases
```

## Configuration

`docker-compose.yml` loads `.env` via `env_file`; all variables are read by
`dietary_advisor.config.Settings`, plus setup-only knobs (source URLs, download
caches, RAG chunking) in `setup.settings.SetupSettings`; evaluation output paths live in
`evaluation.settings.EvaluationSettings` — both extend the shared `Settings`.

`DA_LLM_MODEL`/`DA_MEAL_IDEA_LLM_MODEL` are `pydantic-ai` model identifiers
(`openai:gpt-4o-mini`, `anthropic:claude-3-5-sonnet-latest`, `groq:llama-3.3-70b-versatile`,
...). The meal-idea brainstorm runs on `DA_MEAL_IDEA_LLM_MODEL` independently of
`DA_LLM_MODEL` (and of the model the ablation harness sweeps). Soft-preference scoring
uses two fixed judges (see `evaluation/judges.py`) with the same rubric; each judge scores every run
`JUDGE_REPS` times into `outputs/scores_judge_N/{spec_key}__jrep{n}.json`, and all reporting uses the
mean verdict across those repetitions.

`DA_LLM_MODEL` may carry a `#<effort>` suffix (`minimal`/`low`/`medium`/`high`/`xhigh`,
e.g. `openrouter:openai/gpt-5.6-luna#high`) to request a non-default reasoning
effort - see `dietary_advisor.config.LlmSpec`. Without a suffix, the provider's
own default applies. Local models served through an OpenAI-compatible endpoint (e.g.
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