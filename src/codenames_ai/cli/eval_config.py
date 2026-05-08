from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GameZipf(BaseModel):
    """Zipf frequency window for board-card (game) vocabulary."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(default=4.0, description="Zipf minimum for game words")
    max: float = Field(default=6.5, description="Zipf maximum for game words")


class ClueZipf(BaseModel):
    """Zipf frequency window for clue vocabulary."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(default=3.0, description="Zipf minimum for clue words")
    max: float = Field(default=7.0, description="Zipf maximum for clue words")


class EvalAgentConfigFile(BaseModel):
    """YAML schema for `codenames-ai eval --config file.yaml`."""

    model_config = ConfigDict(extra="forbid")

    label: str = "default"
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    language: str = "en"

    game_zipf: GameZipf = Field(default_factory=GameZipf)
    clue_zipf: ClueZipf = Field(default_factory=ClueZipf)
    game_allowed_pos: list[str] = Field(
        default_factory=lambda: ["NOUN"],
    )
    clue_allowed_pos: list[str] = Field(
        default_factory=lambda: ["NOUN", "ADJ", "VERB"],
    )
    exclusions_path: Path | None = None

    top_k_trace: int = Field(
        default=200,
        ge=1,
        le=10_000,
        description="Max candidates kept in SpymasterTrace after rerank.",
    )
    llm_rerank: bool = True
    embedding_top_k: int = Field(
        default=20,
        ge=1,
        le=10_000,
        description="Top embedding-scored candidates sent to the spymaster LLM.",
    )
    blend_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    guesser_extra_candidates: int = 3

    prefer_min_targets: int = Field(
        default=3,
        ge=1,
        le=9,
        description="Soft min targets per clue (capped by friendlies on board).",
    )
    expected_reward_weight: float = Field(
        default=1.10,
        ge=0.0,
        le=5.0,
        description="Weight of Monte Carlo expected reward in spymaster scoring.",
    )
    mc_trials: int = Field(
        default=96,
        ge=1,
        le=2000,
        description="Monte Carlo rollouts per `(clue, N)` scoring candidate.",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_flat_keys(cls, data: Any) -> Any:
        """Accept deprecated flat *_zipf_* keys and rerank_top_k YAML keys."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "game_zipf" not in d and ("game_zipf_min" in d or "game_zipf_max" in d):
            d["game_zipf"] = {
                "min": d.pop("game_zipf_min", 4.0),
                "max": d.pop("game_zipf_max", 6.5),
            }
        if "clue_zipf" not in d and ("clue_zipf_min" in d or "clue_zipf_max" in d):
            d["clue_zipf"] = {
                "min": d.pop("clue_zipf_min", 3.0),
                "max": d.pop("clue_zipf_max", 7.0),
            }
        if "embedding_top_k" not in d and "rerank_top_k" in d:
            d["embedding_top_k"] = d.pop("rerank_top_k")
        elif "rerank_top_k" in d:
            d.pop("rerank_top_k")
        return d


def load_eval_yaml(path: Path) -> tuple[EvalAgentConfigFile, str]:
    """Load and validate a YAML file; return `(config, config_hash_hex16)`."""
    import hashlib

    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    data = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    cfg = EvalAgentConfigFile.model_validate(data)
    if cfg.exclusions_path is not None and not cfg.exclusions_path.is_absolute():
        rel = path.parent / cfg.exclusions_path
        cfg = cfg.model_copy(update={"exclusions_path": rel})
    return cfg, digest
