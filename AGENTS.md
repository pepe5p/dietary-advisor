# AGENTS.md

Guidance for coding agents working in this repository.

## Package layering

The repo has three top-level Python packages with a strict, one-directional
dependency rule:

- **`dietary_advisor/`** — the runtime application (the LLM agent, food DB
  reader, totaller, RAG retrieval, reflection). It must contain **only code
  needed to run the app**. No build/provisioning logic, no evaluation logic.
- **`setup/`** — one-off build/provisioning code (downloading the Open Food
  Facts export, building the DuckDB, embedding product documents, ingesting the
  RAG corpus). Runs via `just setup`, never in the request path.
- **`evaluation/`** — the ablation harness (case collection via `just
  run-cases`, scoring via `just evaluate`). Never in the request path.
- **`repl/`** — interactive debugging helpers loaded by `ipython_startup.py`
  (`just ps`). One module per concern (e.g. `food_db.py`); each module prints
  its own manual table on import via `repl/manual.py`. Free to reach into
  private internals of `dietary_advisor` readers that the runtime deliberately
  keeps unexported (e.g. per-channel search methods), since it exists purely
  for ad hoc inspection.

Allowed imports: `setup`, `evaluation`, and `repl` may import from
`dietary_advisor`. `dietary_advisor` must **never** import from `setup`,
`evaluation`, or `repl`.

## Agent contract: no invented foods

The nutrition agent (`dietary_advisor/agents/nutrition/agent.py`) never emits
nutrient values and cannot invent a food. Its structured output
(`AgentMealPlan`, `dietary_advisor/agents/agent_output.py`) references foods
only by `PortionRef(code, name, grams)`, where `code` must come from a
`lookup_foods` hit. An output validator (`_validate_codes`)
resolves every code against the real `FoodDb` before accepting the run,
raising `ModelRetry` on anything that doesn't exist. `dietary_advisor.planning.hydration`
is the only place a reference resolves into a real, DB-verified `FoodItem`
(`to_food_item`/`hydrate_meal_plan`), used identically by the production
pipeline and the evaluation harness. Because of this, the food DB and its
lookup tools are always on and are not part of `VariantConfig`/the ablation
grid (see the docstring in `dietary_advisor/planning/pipeline.py`).

When a piece of logic is shared between runtime and a build/eval step, keep the
shared primitive in `dietary_advisor` and call it from the outer package — do
not pull build-only helpers into `dietary_advisor`. Example: the fastembed
model loader (`dietary_advisor/food_db/embeddings.py:load_embedder`) is shared,
but the build-time document embedder lives in
`setup/embedding.py` (`embed_documents`) while only the runtime query
embedder (`embed_query`) stays in `dietary_advisor`.
