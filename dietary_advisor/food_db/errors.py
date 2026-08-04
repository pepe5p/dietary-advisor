"""Unknown-food-code exceptions raised by the OFF/USDA readers and the `FoodDb` facade."""

from typing import Any


class UnknownFoodCodeError(Exception):
    """A single food code that does not resolve against a food database."""

    def __init__(self, code: str, *args: Any) -> None:
        self.code = code
        super().__init__(*args or (f"unknown food code: {code!r}",))


class MultipleUnknownFoodCodesError(Exception):
    """Every unresolved code found while resolving one plan.

    `codes` is deduplicated and sorted so a caller can log the complete set in
    one line instead of only the first miss.
    """

    def __init__(self, codes: list[str], *args: Any) -> None:
        self.codes = sorted(set(codes))
        super().__init__(*args or (f"unknown food code(s): {self.codes}",))


class OFFUnknownFoodCodeError(UnknownFoodCodeError):
    """A code that does not resolve against the Open Food Facts database."""


class USDAUnknownFoodCodeError(UnknownFoodCodeError):
    """A code that does not resolve against the USDA database."""
