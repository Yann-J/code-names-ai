from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class EmbeddingProvider(ABC):
    """Maps surface forms to dense vectors.

    Implementations are responsible for handling out-of-vocabulary inputs in
    whatever way fits the provider (e.g. fastText returns subword-aggregated
    vectors; a sentence-transformer would just embed the string).
    """

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable identifier used in artifact filenames; should change when the
        underlying model changes."""

    @abstractmethod
    def vectorize(self, surfaces: list[str]) -> "np.ndarray":
        """Return an `(N, dim)` float32 array for the given surface forms."""


class FastTextProvider(EmbeddingProvider):
    """`EmbeddingProvider` backed by a Facebook fastText `.bin` model.

    The model is loaded lazily on first use; the file path is the source of
    truth for the provider's identity.
    """

    def __init__(self, model_path: Path, *, identifier: str | None = None) -> None:
        self._model_path = Path(model_path)
        self._identifier = identifier or f"fasttext-{self._model_path.stem}"
        self._model = None
        self._dim: int | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def provider_id(self) -> str:
        return self._identifier

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def vectorize(self, surfaces: list[str]) -> "np.ndarray":
        import numpy as np

        self._ensure_loaded()
        assert self._model is not None
        vectors = np.empty((len(surfaces), self.dim), dtype=np.float32)
        for i, surface in enumerate(surfaces):
            vectors[i] = self._model.get_word_vector(surface)
        return vectors

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"fastText model not found at {self._model_path}.\n"
                f"Download with: codenames-ai download fasttext --lang <lang>"
            )
        try:
            import fasttext
        except ImportError as e:
            raise RuntimeError(
                "The 'fasttext' package is required. Install with: pip install fasttext-wheel"
            ) from e
        self._model = fasttext.load_model(str(self._model_path))
        self._dim = self._model.get_dimension()
