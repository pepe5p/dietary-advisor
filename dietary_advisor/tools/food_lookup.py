"""Protocol for resolving canonical :class:`~dietary_advisor.schemas.nutrition.FoodItem` by ``fdc_id``."""

from __future__ import annotations

from typing import Protocol

from dietary_advisor.schemas.nutrition import FoodItem


class FoodLookup(Protocol):
    """Lookup interface implemented by :class:`~dietary_advisor.tools.usda_client.USDAClient`."""

    def get_food(self, fdc_id: int) -> FoodItem:
        """Return the cached (or fetched) food entry for ``fdc_id``."""
        ...
