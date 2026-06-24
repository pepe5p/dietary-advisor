"""Entry point: ``python -m setup`` proxies to the Typer CLI."""

from setup.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
