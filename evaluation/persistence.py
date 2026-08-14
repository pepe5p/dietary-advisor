"""Shared path and JSON persistence helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from evaluation.settings import get_evaluation_settings

if TYPE_CHECKING:
    from evaluation.case_runner.grid import RunSpec

T = TypeVar("T", bound=BaseModel)


def resolve_output_dir(output_dir: Path | None) -> Path:
    return output_dir if output_dir is not None else get_evaluation_settings().output_dir


def filename_for(spec: RunSpec) -> str:
    return f"{spec.spec_key}.json"


def artifact_path(spec: RunSpec, *, output_dir: Path | None = None, subdir: str) -> Path:
    return resolve_output_dir(output_dir) / subdir / filename_for(spec)


def is_present(path: Path) -> bool:
    return path.is_file()


def save_json(path: Path, record: BaseModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_json(path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))
