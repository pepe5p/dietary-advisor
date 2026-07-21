"""Runtime text embedder for semantic product search.

Only the query side lives here (runtime code). The build-time document
embedder is in `setup.embedding`, but both must share `load_embedder` so their
vectors land in the same space.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from dietary_advisor.config import get_settings

log = logging.getLogger(__name__)

# BGE-M3 is not in fastembed's built-in registry, so it must be registered as a
# custom model first. CLS pooling + L2 normalization match the model card;
# the ONNX weights (plus external data blob) ship under onnx/ in the HF repo.
_BGE_M3 = "BAAI/bge-m3"


def _register_custom_models(dim: int) -> None:
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if _BGE_M3 in supported:
        return
    TextEmbedding.add_custom_model(
        model=_BGE_M3,
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf=_BGE_M3),
        dim=dim,
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    )


@lru_cache(maxsize=1)
def load_embedder() -> TextEmbedding:
    """The shared fastembed model (loaded once), used by query and document sides alike."""
    settings = get_settings()
    _register_custom_models(settings.off_embedding_dim)
    log.info("Loading fastembed model %r (cache: %s)", settings.off_embedding_model, settings.off_embedding_cache)
    return TextEmbedding(
        model_name=settings.off_embedding_model,
        cache_dir=str(settings.off_embedding_cache),
    )


def embed_query(text: str) -> list[float]:
    """Embed a search query."""
    return next(iter(load_embedder().embed([text]))).tolist()
