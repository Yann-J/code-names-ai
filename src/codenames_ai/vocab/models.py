from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class VocabConfig:
    """Inputs that determine the contents of a built `Vocabulary`.

    Two `VocabConfig` instances with equal `cache_key()` produce identical
    artifacts; the cache key is computed from every field that influences
    the build (including the contents of the exclusions file).
    """

    language: str
    zipf_min: float
    zipf_max: float
    allowed_pos: frozenset[str]
    min_length: int = 3
    allow_hyphens: bool = True
    exclusions_path: Path | None = None

    def cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(self.language.encode())
        h.update(f"{self.zipf_min:.4f}".encode())
        h.update(f"{self.zipf_max:.4f}".encode())
        for pos in sorted(self.allowed_pos):
            h.update(pos.encode())
        h.update(str(self.min_length).encode())
        h.update(str(self.allow_hyphens).encode())
        if self.exclusions_path is not None and self.exclusions_path.exists():
            h.update(self.exclusions_path.read_bytes())
        return h.hexdigest()[:12]


_VOCAB_COLUMNS = ["surface", "lemma", "zipf", "pos"]


@dataclass(frozen=True)
class Vocabulary:
    """A built vocabulary artifact: a DataFrame plus the config that produced it."""

    config: VocabConfig
    df: "pd.DataFrame" = field(repr=False)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def surfaces(self) -> list[str]:
        return self.df["surface"].tolist()

    @property
    def lemmas(self) -> list[str]:
        return self.df["lemma"].tolist()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, index=False)

    @classmethod
    def load(cls, config: VocabConfig, path: Path) -> Vocabulary:
        import pandas as pd

        df = pd.read_parquet(path)
        missing = set(_VOCAB_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Vocabulary parquet at {path} missing columns: {sorted(missing)}")
        return cls(config=config, df=df[_VOCAB_COLUMNS].reset_index(drop=True))
