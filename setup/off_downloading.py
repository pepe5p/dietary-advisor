"""Fetch the Open Food Facts Parquet export to a local cache.

The full export is ~7.6 GB. We download it once to a local cache and then
filter from that file (see ``setup.duckdb_creation``), rather than streaming it
over ``httpfs`` on every run: DuckDB's remote Parquet reads fan out into
hundreds of ranged HTTP requests, which Hugging Face rate-limits (HTTP 429). A
single resumable download is both friendlier to HF and far faster to re-run.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

from dietary_advisor.config import Settings

log = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
_DOWNLOAD_LOG_EVERY_BYTES = 200 << 20  # log progress roughly every 200 MiB
_DOWNLOAD_TIMEOUT_S = 120.0
_DOWNLOAD_RETRIES = 5


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GiB"


def _download_parquet(url: str, dest: Path) -> None:
    """Download `url` to `dest`, resuming a prior partial download when possible.

    Bytes accumulate in a sibling ``.part`` file (via HTTP Range requests) and
    are renamed onto `dest` only once complete, so an interrupted run never
    leaves a truncated file masquerading as the finished download. Transient
    failures (including HF's 429) are retried with exponential backoff.
    """
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download from non-https URL: {url!r}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "dietary-advisor-setup"})  # noqa: S310 (https-only)
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")
        try:
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:  # noqa: S310 (https-only)
                # A 200 (rather than 206) means the server ignored our Range
                # header - e.g. after a redirect dropped it - so start over.
                if resume_from and resp.status != 206:
                    resume_from = 0
                content_length = resp.headers.get("Content-Length")
                total = (int(content_length) + resume_from) if content_length else None
                downloaded = resume_from
                since_log = 0
                mode = "ab" if resume_from else "wb"
                if resume_from:
                    log.info("Resuming download at %s...", _human_bytes(resume_from))
                with part.open(mode) as fh:
                    while True:
                        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        since_log += len(chunk)
                        if since_log >= _DOWNLOAD_LOG_EVERY_BYTES:
                            since_log = 0
                            if total:
                                log.info(
                                    "  downloaded %s / %s (%.0f%%)",
                                    _human_bytes(downloaded),
                                    _human_bytes(total),
                                    100 * downloaded / total,
                                )
                            else:
                                log.info("  downloaded %s", _human_bytes(downloaded))
            part.replace(dest)
            log.info("Download complete: %s (%s).", dest, _human_bytes(dest.stat().st_size))
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            wait = min(2**attempt, 30)
            log.warning("Download attempt %d/%d failed: %s. Retrying in %ds.", attempt, _DOWNLOAD_RETRIES, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to download OFF Parquet from {url} after {_DOWNLOAD_RETRIES} attempts: {last_exc}")


def resolve_source_parquet(settings: Settings) -> Path:
    """Return a local Parquet path to filter, downloading the OFF export to the cache if needed.

    Uses the cache at `settings.off_raw_parquet`, downloading it from
    `settings.off_source_url` when missing.
    """
    cache = settings.off_raw_parquet
    if cache.exists() and cache.stat().st_size > 0:
        log.info("Using cached OFF Parquet at %s (%s).", cache, _human_bytes(cache.stat().st_size))
        return cache

    log.info("Downloading OFF Parquet (~7.6 GB) from %s to %s ...", settings.off_source_url, cache)
    _download_parquet(settings.off_source_url, cache)
    return cache
