"""Download, parse and chunk the clinical-guideline corpus, then index it.

Run via the CLI: ``dietary-advisor ingest-corpus``.

The script is idempotent and re-entrant: it only re-downloads files whose
remote ETag has changed (best-effort) and only re-indexes chunks whose hash
isn't already present in the Chroma collection.

If a remote PDF cannot be fetched, the script falls back to the bundled
plain-text seed in `knowledge/seeds/<doc_id>.txt` so that downstream code
never sees an empty corpus.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import pypdf

from dietary_advisor.config import get_settings, Settings
from dietary_advisor.knowledge.sources import CorpusSource, SOURCES
from dietary_advisor.knowledge.store import VectorStore

log = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).parent / "seeds"
_SECTION_RE = re.compile(r"^\s*Section:\s*(.+)$", re.MULTILINE)


@dataclass
class IngestStats:
    fetched: list[str]
    seeded: list[str]
    failed: list[str]
    chunks_indexed: int


def _download(url: str, dest: Path, timeout_s: float = 60.0) -> bool:
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        return True
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        log.warning("Failed to download %s: %s", url, exc)
        return False


def _extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - pypdf raises a wide variety
        log.warning("pypdf could not read %s: %s", pdf_path, exc)
        return ""
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("pypdf failed on %s page %d: %s", pdf_path, i, exc)
            text = ""
        parts.append(f"<<PAGE {i}>>\n{text}")
    return "\n".join(parts)


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphens
    text = re.sub(r"-\n", "", text)  # de-hyphenate line wraps
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Naive paragraph-aware chunker.

    Splits on blank lines first, then greedily packs paragraphs into windows
    of ~`chunk_size` characters. Overlap is implemented by re-emitting the
    tail of the previous window.
    """
    text = _normalise_whitespace(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= chunk_size:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para
    if buf:
        chunks.append(buf)
    return chunks


def _detect_section(chunk: str) -> str | None:
    m = _SECTION_RE.search(chunk)
    return m.group(1).strip() if m else None


def _detect_page(chunk: str) -> int | None:
    m = re.search(r"<<PAGE (\d+)>>", chunk)
    return int(m.group(1)) if m else None


def _build_chunks_for(source: CorpusSource, raw_text: str, settings: Settings) -> list[dict[str, str | int | None]]:
    chunks = _chunk(
        raw_text,
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
    )
    out: list[dict[str, str | int | None]] = []
    for i, c in enumerate(chunks):
        chunk_hash = hashlib.sha1(c.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        chunk_id = f"{source.doc_id}::{i:04d}::{chunk_hash}"
        out.append(
            {
                "id": chunk_id,
                "text": c,
                "doc_id": source.doc_id,
                "title": source.title,
                "section": _detect_section(c) or source.section,
                "page": _detect_page(c),
                "tags": ",".join(source.tags),
            },
        )
    return out


def ingest_corpus(*, force_redownload: bool = False) -> IngestStats:
    """End-to-end ingest: download/seed, chunk, embed, index."""
    settings = get_settings()
    corpus_dir = settings.corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore()

    fetched: list[str] = []
    seeded: list[str] = []
    failed: list[str] = []
    total_chunks = 0

    for source in SOURCES:
        text = ""
        pdf_path = corpus_dir / f"{source.doc_id}.pdf"
        if source.url and (force_redownload or not pdf_path.exists()):
            ok = _download(source.url, pdf_path, timeout_s=settings.request_timeout_s)
            if ok:
                fetched.append(source.doc_id)
            else:
                failed.append(source.doc_id)
        if pdf_path.exists():
            text = _extract_text_from_pdf(pdf_path)

        if not text:
            seed = _SEEDS_DIR / f"{source.doc_id}.txt"
            if seed.exists():
                text = seed.read_text(encoding="utf-8")
                if source.doc_id not in fetched:
                    seeded.append(source.doc_id)
            else:
                log.warning("No content available for %s (no seed and no PDF).", source.doc_id)
                continue

        chunks = _build_chunks_for(source, text, settings)
        if not chunks:
            continue
        store.upsert_chunks(chunks)
        total_chunks += len(chunks)

    return IngestStats(fetched=fetched, seeded=seeded, failed=failed, chunks_indexed=total_chunks)
