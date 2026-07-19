"""Dietary RAG: clinical guidelines retrieval (WHO, NICE, ADA, USDA DGA)."""

from dietary_advisor.dietary_rag.retriever import HybridRetriever, RetrievedChunk
from dietary_advisor.dietary_rag.store import Chunk, ChunkMeta, QueryHit, VectorStore

__all__ = [
    "Chunk",
    "ChunkMeta",
    "HybridRetriever",
    "QueryHit",
    "RetrievedChunk",
    "VectorStore",
]
