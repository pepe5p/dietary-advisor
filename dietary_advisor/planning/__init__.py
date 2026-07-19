"""Plan assembly: the orchestrator, hydration, and shopping-list building.

Deliberately has no re-exports: `pipeline` imports the `agents` package, which
in turn imports `hydration` (via `nutrition_agent`/`reflection`), so a
re-exporting `__init__` that pulls in `pipeline` at package-import time would
risk a circular import. Import submodules directly, e.g.
`from dietary_advisor.planning.pipeline import Pipeline`.
"""

from __future__ import annotations
