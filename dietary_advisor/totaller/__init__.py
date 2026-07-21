"""Totaller: deterministic nutrient-summation tool exposed to the agent.

Deliberately has no re-exports: `aggregate` imports `planning.meal_plan`, which
imports `totaller.nutrition`, so a re-exporting `__init__` that pulls in
`aggregate` at package-import time would risk a circular import. Import
submodules directly, e.g.
`from dietary_advisor.totaller.aggregate import total_meal_plan`.
"""

from __future__ import annotations
