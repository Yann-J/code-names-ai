from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codenames_ai.config import Config


@dataclass(frozen=True)
class StoragePaths:
    """Resolved on-disk locations for cached artifacts.

    All paths are content-addressed by config hash at the call sites that
    produce artifacts; this class only owns the directory layout.
    """

    cache_dir: Path

    @classmethod
    def from_config(cls, config: Config) -> StoragePaths:
        return cls(cache_dir=config.cache_dir)

    @property
    def vocab_dir(self) -> Path:
        return self.cache_dir / "vocab"

    @property
    def embed_dir(self) -> Path:
        return self.cache_dir / "embed"

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def evals_dir(self) -> Path:
        return self.cache_dir / "evals"

    @property
    def llm_cache_path(self) -> Path:
        return self.cache_dir / "llm.sqlite"

    def vocab_dir_for(self, language: str) -> Path:
        return self.vocab_dir / language

    def embed_dir_for(self, language: str) -> Path:
        return self.embed_dir / language

    def ensure(self) -> None:
        for d in (self.vocab_dir, self.embed_dir, self.models_dir, self.evals_dir):
            d.mkdir(parents=True, exist_ok=True)
