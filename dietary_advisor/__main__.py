"""Entry point: ``python -m dietary_advisor`` proxies to the Typer CLI."""

from dietary_advisor.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
