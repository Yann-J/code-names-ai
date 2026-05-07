from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EvalAgentConfigFile(BaseModel):
    """YAML schema for `codenames-ai eval --config file.yaml`."""

    label: str = "default"
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    language: str = "en"

    game_zipf_min: float = 4.0
    game_zipf_max: float = 6.5
    clue_zipf_min: float = 3.0
    clue_zipf_max: float = 7.0
    game_allowed_pos: list[str] = Field(default_factory=lambda: ["NOUN"])
    clue_allowed_pos: list[str] = Field(default_factory=lambda: ["NOUN", "ADJ"])
    exclusions_path: Path | None = None

    top_k_trace: int = 50
    llm_rerank: bool = True
    rerank_top_k: int = 10
    blend_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    guesser_extra_candidates: int = 3


def load_eval_yaml(path: Path) -> tuple[EvalAgentConfigFile, str]:
    """Load and validate a YAML file; return `(config, config_hash_hex16)`."""
    import hashlib

    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    data = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    cfg = EvalAgentConfigFile.model_validate(data)
    if cfg.exclusions_path is not None and not cfg.exclusions_path.is_absolute():
        cfg = cfg.model_copy(update={"exclusions_path": path.parent / cfg.exclusions_path})
    return cfg, digest
