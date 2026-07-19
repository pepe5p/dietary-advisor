"""Unknown-food-code exceptions raised by the OFF/USDA readers and the `FoodDb` facade."""

from __future__ import annotations


class UnknownFoodCodeError(Exception):
    """A food code that does not resolve against any food database."""


class OFFUnknownFoodCodeError(UnknownFoodCodeError):
    """A code that does not resolve against the Open Food Facts database."""


class USDAUnknownFoodCodeError(UnknownFoodCodeError):
    """A code that does not resolve against the USDA database."""
