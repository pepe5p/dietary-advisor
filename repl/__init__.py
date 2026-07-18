"""REPL-only debug helpers (loaded via `PYTHONSTARTUP=ipython_startup.py`).

Not part of the runtime, build, or evaluation packages - see the layering
rule in AGENTS.md. Each module here is free to reach into private internals
of `dietary_advisor` readers (e.g. per-channel search methods) that the
runtime deliberately keeps unexported, and prints its own manual on import.
"""
