from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.embedding.provider import EmbeddingProvider
from codenames_ai.storage import StoragePaths
from codenames_ai.vocab.models import Vocabulary

logger = logging.getLogger(__name__)


def build_embedding_matrix(
    vocabulary: Vocabulary, provider: EmbeddingProvider
) -> EmbeddingMatrix:
    """Project every word in `vocabulary` through `provider`, L2-normalize, and return."""
    surfaces = vocabulary.surfaces
    if not surfaces:
        return EmbeddingMatrix(
            vectors=np.zeros((0, provider.dim), dtype=np.float32),
            surfaces=[],
            surface_to_index={},
            provider_id=provider.provider_id,
            vocab_cache_key=vocabulary.config.cache_key(),
        )

    logger.info("projecting %d words through %s", len(surfaces), provider.provider_id)
    vectors = provider.vectorize(surfaces).astype(np.float32, copy=False)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against zero-norm rows (e.g. fastText returning all zeros for empty input).
    norms = np.where(norms == 0, 1.0, norms)
    normalized = (vectors / norms).astype(np.float32)

    return EmbeddingMatrix(
        vectors=normalized,
        surfaces=list(surfaces),
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id=provider.provider_id,
        vocab_cache_key=vocabulary.config.cache_key(),
    )


def load_or_build_embedding_matrix(
    vocabulary: Vocabulary,
    provider: EmbeddingProvider,
    storage: StoragePaths,
) -> EmbeddingMatrix:
    """Load a cached embedding matrix if available; otherwise build, save, and return one."""
    cache_path = _matrix_cache_path(vocabulary, provider, storage)
    if cache_path.exists():
        logger.info("embedding matrix cache hit: %s", cache_path)
        return EmbeddingMatrix.load(cache_path)

    logger.info("embedding matrix cache miss; building")
    matrix = build_embedding_matrix(vocabulary, provider)
    matrix.save(cache_path)
    return matrix


def _matrix_cache_path(
    vocabulary: Vocabulary, provider: EmbeddingProvider, storage: StoragePaths
):
    vocab_key = vocabulary.config.cache_key()
    provider_key = _provider_cache_key(provider.provider_id)
    return (
        storage.embed_dir_for(vocabulary.config.language)
        / f"{vocab_key}__{provider_key}.npz"
    )


def _provider_cache_key(provider_id: str) -> str:
    """Short, filename-safe digest of a provider id."""
    digest = hashlib.sha256(provider_id.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", provider_id)[:40].strip("-") or "provider"
    return f"{safe}-{digest}"
