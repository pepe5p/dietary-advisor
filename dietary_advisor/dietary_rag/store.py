"""ChromaDB wrapper. Persistent, embedded, no Docker required.

By default Chroma uses its bundled ONNX-based MiniLM-L6-v2 embedding function
(`DefaultEmbeddingFunction`), which keeps the system fully self-contained. The
embedding function is configurable via the optional `embedding_fn` argument so
the ablation study can swap to OpenAI embeddings if desired.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Where
from chromadb.utils import embedding_functions
from pydantic import BaseModel

from dietary_advisor.config import get_settings

log = logging.getLogger(__name__)

_COLLECTION = "guidelines"


class ChunkMeta(BaseModel):
    """Per-chunk metadata that round-trips through Chroma's metadata store.

    Pydantic because Chroma persists a flat mapping and returns it untyped;
    `model_validate` restores `page` as an int (Chroma keeps the type, but the
    model is the single place that contract is stated).
    """

    doc_id: str
    title: str
    page: int | None = None


@dataclass
class Chunk:
    """A corpus chunk ready to index, and the element type of `all_documents`."""

    id: str
    text: str
    meta: ChunkMeta


@dataclass
class QueryHit:
    id: str
    text: str
    meta: ChunkMeta
    distance: float | None


class VectorStore:
    """Persistent vector index keyed by chunk id."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_fn: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        settings = get_settings()
        self._dir = persist_dir or settings.chroma_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._dir))
        if embedding_fn is None:
            # Chroma hardcodes the bundled MiniLM download under $HOME, which is
            # ephemeral in the container. Point it at settings.onnx_model_dir
            # (on the bind-mounted data dir) so the ~80 MB model is fetched once
            # and survives restarts. DefaultEmbeddingFunction builds a fresh
            # ONNXMiniLM_L6_V2 per call, so the redirect has to live on the class.
            onnx_ef = embedding_functions.ONNXMiniLM_L6_V2
            onnx_ef.DOWNLOAD_PATH = settings.onnx_model_dir / onnx_ef.MODEL_NAME
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._embedding_fn = embedding_fn
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.meta.model_dump(exclude_none=True) for c in chunks]
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)  # type: ignore[arg-type]

    def query(self, query: str, top_k: int = 6) -> list[QueryHit]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[QueryHit] = []
        for i, doc in enumerate(docs):
            out.append(
                QueryHit(
                    id=ids[i],
                    text=doc,
                    meta=ChunkMeta.model_validate(metas[i] or {}),
                    distance=float(dists[i]) if i < len(dists) else None,
                ),
            )
        log.debug("vector_store.query(%r, top_k=%d) -> %d hit(s)", query, top_k, len(out))
        return out

    def delete_docs(self, doc_ids: Sequence[str]) -> None:
        """Drop every chunk belonging to the given source documents.

        `upsert_chunks` never removes anything, so a source dropped from the
        curated list would otherwise linger in the index; the setup step calls
        this to purge chunks whose `doc_id` is no longer part of the corpus.
        """
        if not doc_ids:
            return
        where = cast("Where", {"doc_id": {"$in": list(doc_ids)}})
        self._collection.delete(where=where)

    def all_documents(self) -> list[Chunk]:
        """Materialise the entire collection - used to seed the BM25 index."""
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [Chunk(id=ids[i], text=doc, meta=ChunkMeta.model_validate(metas[i] or {})) for i, doc in enumerate(docs)]

    def count(self) -> int:
        return int(self._collection.count())
