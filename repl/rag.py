"""REPL helpers for exploring the dietary RAG retrieval pipeline.

Includes per-channel (`_bm25`/`_dense`) retrieval variants that reach into
the retriever's private scorers to bypass the hybrid fusion - useful for
debugging a query where fusion buries one channel's best hit.
"""

from __future__ import annotations

from rich.table import Table

from dietary_advisor.dietary_rag.retriever import HybridRetriever, RetrievedChunk
from dietary_advisor.dietary_rag.store import Chunk
from dietary_advisor.planning.meal_plan import Citation
from repl.manual import console, print_manual

__all__ = [
    "citations",
    "corpus_stats",
    "pchunk",
    "prc",
    "retrieve",
    "retrieve_bm25",
    "retrieve_dense",
]

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Return a process-wide `HybridRetriever`, built once.

    The constructor materialises the whole Chroma collection to seed its BM25
    index, so rebuilding it on every call would make each REPL query pay that
    cost again.
    """
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _resolve(chunk_id: str, docs: list[Chunk]) -> Chunk | None:
    for doc in docs:
        if doc.id == chunk_id:
            return doc
    return None


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    return _get_retriever().retrieve(query, top_k=top_k)


def retrieve_bm25(query: str, top_k: int = 10) -> list[RetrievedChunk]:
    retriever = _get_retriever()
    scores = retriever._bm25_scores(query)  # noqa: SLF001
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out: list[RetrievedChunk] = []
    for chunk_id, score in ranked:
        doc = _resolve(chunk_id, retriever._docs)  # noqa: SLF001
        if doc is None:
            continue
        out.append(RetrievedChunk(id=chunk_id, text=doc.text, meta=doc.meta, score=round(score, 4)))
    return out


def retrieve_dense(query: str, top_k: int = 10) -> list[RetrievedChunk]:
    retriever = _get_retriever()
    hits = retriever._dense_scores(query, top_k)  # noqa: SLF001
    out = [
        RetrievedChunk(id=chunk_id, text=hit.text, meta=hit.meta, score=round(max(0.0, 1.0 - (hit.distance or 0.0)), 4))
        for chunk_id, hit in hits.items()
    ]
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_k]


def citations(query: str, top_k: int | None = None) -> list[Citation]:
    return _get_retriever().retrieve_citations(query, top_k=top_k)


def corpus_stats() -> None:
    store = _get_retriever()._store  # noqa: SLF001
    docs = store.all_documents()
    counts: dict[tuple[str, str], int] = {}
    for doc in docs:
        key = (doc.meta.doc_id, doc.meta.title)
        counts[key] = counts.get(key, 0) + 1

    table = Table(title=f"Corpus: {store.count()} chunk(s)", show_lines=True, header_style="bold cyan")
    table.add_column("doc_id", style="bold")
    table.add_column("title", overflow="fold")
    table.add_column("chunks", justify="right")
    for (doc_id, title), count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        table.add_row(doc_id, title, str(count))
    console.print(table)


def _snippet(text: str, max_chars: int = 160) -> str:
    snippet = text.strip().replace("\n", " ")
    return snippet if len(snippet) <= max_chars else snippet[:max_chars].rstrip() + "..."


def prc(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        console.print("[yellow]No chunks retrieved.[/yellow]")
        return
    table = Table(title="Retrieved chunks", show_lines=True, header_style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("doc_id", style="bold")
    table.add_column("Page", justify="right")
    table.add_column("Snippet", overflow="fold")
    for chunk in chunks:
        page = str(chunk.meta.page) if chunk.meta.page is not None else "-"
        table.add_row(f"{chunk.score:.4f}", chunk.meta.doc_id, page, _snippet(chunk.text))
    console.print(table)


def pchunk(chunk: RetrievedChunk) -> None:
    table = Table(title=f"Chunk: {chunk.id}", show_lines=True, header_style="bold cyan")
    table.add_column("Field", style="bold", overflow="fold")
    table.add_column("Value", overflow="fold")
    table.add_row("doc_id", chunk.meta.doc_id)
    table.add_row("title", chunk.meta.title)
    table.add_row("page", str(chunk.meta.page) if chunk.meta.page is not None else "-")
    table.add_row("score", f"{chunk.score:.4f}")
    table.add_row("text", chunk.text)
    console.print(table)


_MANUAL: tuple[tuple[str, str], ...] = (
    ("retrieve(query, top_k=None)", "Full hybrid (weighted BM25 + dense fusion) retrieval -> list[RetrievedChunk]."),
    ("retrieve_bm25(query, top_k=10)", "Retrieve, lexical (BM25) channel only."),
    ("retrieve_dense(query, top_k=10)", "Retrieve, embedding (dense) channel only."),
    ("citations(query, top_k=None)", "Run retrieve() and map hits to Citation, as the agent would cite them."),
    ("corpus_stats()", "Print total indexed chunk count, broken down by doc_id/title."),
    ("prc(chunks)", "Rich-print a list of RetrievedChunk as a table."),
    ("pchunk(chunk)", "Rich-print one RetrievedChunk's full detail."),
)

print_manual("dietary RAG helpers", _MANUAL)
