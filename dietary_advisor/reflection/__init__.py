"""Reflection loop (Pętla Walidacyjna): critique-then-refine self-correction.

A tool-less critic agent reviews the plan (grounded by deterministic macro
totals when the totaller is enabled) and only the refiner agent runs when it
reports issues - see `reflection.py` for the loop. Hard-constraint validation
is an evaluation-only concept and lives in `evaluation.validation` instead -
see that package for the code-based rules and Validator.
"""

from dietary_advisor.reflection.reflection import reflect_and_refine, ReflectionResult

__all__ = [
    "ReflectionResult",
    "reflect_and_refine",
]
