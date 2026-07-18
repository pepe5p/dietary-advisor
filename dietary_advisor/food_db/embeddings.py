"""Runtime text embedder for semantic product search.

Only the query side lives here (runtime code). The build-time document
embedder is in `setup` (see `setup.duckdb_creation`), but both must share
`load_embedder` so their vectors land in the same space; the E5 family is
trained with asymmetric `query:` / `passage:` prefixes, applied per side.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from dietary_advisor.config import get_settings

log = logging.getLogger(__name__)

_QUERY_PREFIX = "query: "

# multilingual-e5-small is not in fastembed's built-in registry (only the
# -large variant is), so it must be registered as a custom model first.
# Pooling/normalization follow the model card; the ONNX weights ship in the
# official HF repo under onnx/model.onnx.
_E5_SMALL = "intfloat/multilingual-e5-small"


def _register_custom_models() -> None:
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if _E5_SMALL in supported:
        return
    TextEmbedding.add_custom_model(
        model=_E5_SMALL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=_E5_SMALL),
        dim=384,
        model_file="onnx/model.onnx",
    )


@lru_cache(maxsize=1)
def load_embedder() -> TextEmbedding:
    """The shared fastembed model (loaded once), used by query and document sides alike."""
    _register_custom_models()
    settings = get_settings()
    log.info("Loading fastembed model %r (cache: %s)", settings.off_embedding_model, settings.off_embedding_cache)
    return TextEmbedding(
        model_name=settings.off_embedding_model,
        cache_dir=str(settings.off_embedding_cache),
    )


def embed_query(text: str) -> list[float]:
    """Embed a search query (with the E5 `query:` prefix)."""
    return next(iter(load_embedder().embed([_QUERY_PREFIX + text]))).tolist()
