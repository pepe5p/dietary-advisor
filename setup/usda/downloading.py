"""Fetch and extract the USDA FoodData Central CSV zip exports to a local cache.

Two fixed-vintage archives (Foundation Foods + the final SR Legacy release) are
downloaded once - reusing the OFF resumable downloader - and unzipped in place.
Each archive expands into a single dated subfolder holding the relational CSVs
(`food.csv`, `food_nutrient.csv`, ...); we return the folder that actually
contains `food.csv` so the build never hard-codes the vintage in the path.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from setup.open_food_facts.downloading import _download_parquet
from setup.settings import SetupSettings

log = logging.getLogger(__name__)


def _find_csv_dir(root: Path) -> Path | None:
    """Return the directory under `root` that holds `food.csv`, if extracted."""
    matches = list(root.rglob("food.csv"))
    return matches[0].parent if matches else None


def _ensure_extracted(zip_path: Path, extract_root: Path) -> Path:
    """Extract `zip_path` into `extract_root` (once) and return its CSV directory."""
    existing = _find_csv_dir(extract_root) if extract_root.exists() else None
    if existing is not None:
        log.info("Using extracted USDA CSVs at %s.", existing)
        return existing

    extract_root.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)

    csv_dir = _find_csv_dir(extract_root)
    if csv_dir is None:
        raise RuntimeError(f"USDA archive {zip_path} did not contain a food.csv after extraction.")
    return csv_dir


def resolve_usda_csv_dirs(settings: SetupSettings) -> list[Path]:
    """Return the CSV directories for the Foundation + SR Legacy datasets.

    Downloads each archive to `settings.usda_raw_dir` when missing and unzips it
    in place; both are small enough (a few MB each) to keep alongside the OFF
    caches without a size concern.
    """
    dirs: list[Path] = []
    for url in (settings.usda_foundation_source_url, settings.usda_sr_legacy_source_url):
        filename = url.rsplit("/", 1)[-1]
        zip_dest = settings.usda_raw_dir / filename
        if zip_dest.exists() and zip_dest.stat().st_size > 0:
            log.info("Using cached USDA archive at %s.", zip_dest)
        else:
            log.info("Downloading USDA dataset from %s to %s ...", url, zip_dest)
            _download_parquet(url, zip_dest)
        dirs.append(_ensure_extracted(zip_dest, settings.usda_raw_dir / zip_dest.stem))
    return dirs
