"""Hybrid retriever: dense (Chroma) + sparse (BM25) with weighted fusion.

Dense embeddings give semantic recall; BM25 adds lexical precision. Both matter
for clinical text, where exact terms (e.g. "hypoglycaemia") are decisive and
embedding models often miss them.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from dietary_advisor.config import get_settings
from dietary_advisor.dietary_rag.store import Chunk, ChunkMeta, QueryHit, VectorStore
from dietary_advisor.schemas.meal_plan import Citation

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    meta: ChunkMeta
    score: float

    def to_citation(self, max_snippet_chars: int = 280) -> Citation:
        snippet = self.text.strip().replace("\n", " ")
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars].rstrip() + "..."
        return Citation(
            source=self.meta.doc_id or self.meta.title or "unknown",
            page=self.meta.page,
            snippet=snippet,
        )


class HybridRetriever:
    """Convex combination of normalised BM25 and dense scores."""

    def __init__(
        self,
        store: VectorStore | None = None,
        bm25_weight: float | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or VectorStore()
        self._bm25_weight = bm25_weight if bm25_weight is not None else settings.rag_bm25_weight
        self._top_k = settings.rag_top_k
        self._refresh_bm25()

    def _refresh_bm25(self) -> None:
        docs = self._store.all_documents()
        self._docs: list[Chunk] = docs
        self._tokenised: list[list[str]] = [_tokenise(d.text) for d in docs]
        self._bm25 = BM25Okapi(self._tokenised) if self._tokenised else None

    def _bm25_scores(self, query: str) -> dict[str, float]:
        if self._bm25 is None:
            return {}
        q_tokens = _tokenise(query)
        if not q_tokens:
            return {}
        scores = self._bm25.get_scores(q_tokens)
        max_score = max(scores) if len(scores) else 0.0
        if max_score <= 0:
            return {}
        return {self._docs[i].id: float(scores[i]) / max_score for i in range(len(self._docs))}

    def _dense_scores(self, query: str, top_k: int) -> dict[str, QueryHit]:
        return {hit.id: hit for hit in self._store.query(query, top_k=top_k)}

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        top_k = top_k or self._top_k
        # Pull a wider pool from each retriever, then fuse.
        pool = max(top_k * 4, 20)
        dense = self._dense_scores(query, pool)
        bm25 = self._bm25_scores(query)

        all_ids = set(dense) | set(bm25)
        fused: dict[str, float] = defaultdict(float)
        for cid in all_ids:
            hit = dense.get(cid)
            # Cosine distance -> similarity in [0, 1]; clamp negative to 0.
            d = max(0.0, 1.0 - (hit.distance or 0.0)) if hit else 0.0
            b = bm25.get(cid, 0.0)
            fused[cid] = (1.0 - self._bm25_weight) * d + self._bm25_weight * b

        # Resolve text + metadata, falling back to BM25-only candidates.
        text_meta: dict[str, tuple[str, ChunkMeta]] = {}
        for cid, hit in dense.items():
            text_meta[cid] = (hit.text, hit.meta)
        for cid in all_ids - text_meta.keys():
            for doc in self._docs:
                if doc.id == cid:
                    text_meta[cid] = (doc.text, doc.meta)
                    break

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for cid, score in ranked:
            found = text_meta.get(cid)
            if found is None or not found[0]:
                continue
            text, meta = found
            out.append(RetrievedChunk(id=cid, text=text, meta=meta, score=round(score, 4)))
        log.debug("retriever.retrieve(%r) -> %d chunk(s)", query, len(out))
        return out

    def retrieve_citations(self, query: str, top_k: int | None = None) -> list[Citation]:
        return [c.to_citation() for c in self.retrieve(query, top_k=top_k)]
