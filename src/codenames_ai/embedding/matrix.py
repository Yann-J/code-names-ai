from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingMatrix:
    """A vocabulary's embeddings, L2-normalized so cosine similarity == dot product.

    Self-contained on disk via `.npz` — no provider is needed at load time. To
    embed words outside the matrix, pass a vector to `nearest` directly (encode
    via the provider first).
    """

    vectors: np.ndarray = field(repr=False)
    surfaces: list[str] = field(repr=False)
    surface_to_index: dict[str, int] = field(repr=False)
    provider_id: str
    vocab_cache_key: str

    def __post_init__(self) -> None:
        if self.vectors.dtype != np.float32:
            raise ValueError(f"vectors must be float32, got {self.vectors.dtype}")
        if self.vectors.shape[0] != len(self.surfaces):
            raise ValueError(
                f"vectors rows ({self.vectors.shape[0]}) != surfaces ({len(self.surfaces)})"
            )

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.surfaces)

    def __contains__(self, surface: str) -> bool:
        return surface in self.surface_to_index

    def __getitem__(self, surface: str) -> np.ndarray:
        idx = self.surface_to_index.get(surface)
        if idx is None:
            raise KeyError(surface)
        return self.vectors[idx]

    def index_of(self, surface: str) -> int | None:
        return self.surface_to_index.get(surface)

    def sim(self, a: str, b: str) -> float:
        """Cosine similarity between two surfaces in the matrix."""
        return float(self[a] @ self[b])

    def sim_matrix(
        self, surfaces_a: list[str], surfaces_b: list[str] | None = None
    ) -> np.ndarray:
        """Pairwise cosine similarities between two surface lists.

        If `surfaces_b` is None, uses `surfaces_a` for both axes (returns a
        square matrix).
        """
        a = self._stack(surfaces_a)
        b = a if surfaces_b is None else self._stack(surfaces_b)
        return a @ b.T

    def nearest(
        self,
        query: str | np.ndarray,
        k: int = 10,
        *,
        exclude: Iterable[str] = (),
    ) -> list[tuple[str, float]]:
        """Return the `k` nearest matrix entries to `query`, sorted desc by similarity.

        `query` may be a surface form already in the matrix or a pre-computed
        vector (will be L2-normalized). The query itself is automatically
        excluded when given as a string.
        """
        if isinstance(query, str):
            idx = self.surface_to_index.get(query)
            if idx is None:
                raise KeyError(f"{query!r} not in matrix")
            vec = self.vectors[idx]
            self_index: int | None = idx
        else:
            vec = np.asarray(query, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                raise ValueError("query vector has zero norm")
            vec = vec / norm
            self_index = None

        sims = self.vectors @ vec

        excluded: set[int] = set()
        if self_index is not None:
            excluded.add(self_index)
        for s in exclude:
            i = self.surface_to_index.get(s)
            if i is not None:
                excluded.add(i)

        if excluded:
            sims = sims.copy()
            for i in excluded:
                sims[i] = -np.inf

        k = min(k, len(self.surfaces) - len(excluded))
        if k <= 0:
            return []

        # argpartition picks top-k unsorted, then we sort just those k.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.surfaces[i], float(sims[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "provider_id": self.provider_id,
                "vocab_cache_key": self.vocab_cache_key,
                "dim": self.dim,
            }
        )
        np.savez_compressed(
            path,
            vectors=self.vectors,
            surfaces=np.array(self.surfaces, dtype=object),
            metadata=np.array(metadata),
        )

    @classmethod
    def load(cls, path: Path) -> EmbeddingMatrix:
        with np.load(path, allow_pickle=True) as data:
            vectors = data["vectors"].astype(np.float32, copy=False)
            surfaces = [str(s) for s in data["surfaces"].tolist()]
            metadata = json.loads(str(data["metadata"]))
        return cls(
            vectors=vectors,
            surfaces=surfaces,
            surface_to_index={s: i for i, s in enumerate(surfaces)},
            provider_id=metadata["provider_id"],
            vocab_cache_key=metadata["vocab_cache_key"],
        )

    def _stack(self, surfaces: list[str]) -> np.ndarray:
        missing = [s for s in surfaces if s not in self.surface_to_index]
        if missing:
            raise KeyError(f"surfaces not in matrix: {missing}")
        indices = [self.surface_to_index[s] for s in surfaces]
        return self.vectors[indices]
