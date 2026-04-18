"""ChromaDB wrapper. Persistent, embedded, no Docker required.

By default Chroma uses its bundled ONNX-based MiniLM-L6-v2 embedding function
(`DefaultEmbeddingFunction`), which keeps the system fully self-contained. The
embedding function is configurable via the optional `embedding_fn` argument so
the ablation study can swap to OpenAI embeddings if desired.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

from dietary_advisor.config import get_settings

_COLLECTION = "guidelines"


class VectorStore:
    """Persistent vector index keyed by chunk id."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_fn: EmbeddingFunction[Any] | None = None,
    ) -> None:
        settings = get_settings()
        self._dir = persist_dir or settings.chroma_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._dir))
        self._embedding_fn = embedding_fn or embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas: list[dict[str, Any]] = []
        for c in chunks:
            meta = {
                "doc_id": c["doc_id"],
                "title": c["title"],
                "section": c.get("section") or "",
                "tags": c.get("tags") or "",
            }
            page = c.get("page")
            if page is not None:
                meta["page"] = int(page)
            metas.append(meta)
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def query(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            out.append(
                {
                    "id": ids[i],
                    "text": doc,
                    "metadata": metas[i] or {},
                    "distance": float(dists[i]) if i < len(dists) else None,
                },
            )
        return out

    def all_documents(self) -> list[dict[str, Any]]:
        """Materialise the entire collection - used to seed the BM25 index."""
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            out.append(
                {
                    "id": ids[i],
                    "text": doc,
                    "metadata": metas[i] or {},
                },
            )
        return out

    def count(self) -> int:
        return int(self._collection.count())
