"""Entry point: ``python -m evaluation`` proxies to the Typer CLI."""

from evaluation.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
