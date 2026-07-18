"""Shared console and manual-table renderer for `repl` modules.

Each `repl` module prints its own manual on import (see `print_manual`), so
the startup script's output grows additively as modules are added rather
than requiring a central registry to stay in sync.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def print_manual(title: str, entries: tuple[tuple[str, str], ...]) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold")
    table.add_column("Description")
    for command, description in entries:
        table.add_row(command, description)
    console.print(table)
