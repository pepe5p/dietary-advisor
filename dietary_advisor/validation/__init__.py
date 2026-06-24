"""Reflection loop (Pętla Walidacyjna): plain Generate-Review-Refine self-correction.

Hard-constraint validation is an evaluation-only concept and lives in
`evaluation.validation` instead - see that package for the code-based rules
and Validator.
"""

from dietary_advisor.validation.reflection import reflect_and_refine, ReflectionResult

__all__ = [
    "ReflectionResult",
    "reflect_and_refine",
]
