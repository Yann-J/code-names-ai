from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ZipfWindow(BaseModel):
    """Zipf frequency window for a vocabulary."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float


class VocabularySideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zipf: ZipfWindow
    allowed_pos: list[str]


class GameVocabularySideConfig(VocabularySideConfig):
    zipf: ZipfWindow = Field(default_factory=lambda: ZipfWindow(min=4.0, max=6.5))
    allowed_pos: list[str] = Field(default_factory=lambda: ["NOUN"])


class ClueVocabularySideConfig(VocabularySideConfig):
    zipf: ZipfWindow = Field(default_factory=lambda: ZipfWindow(min=3.0, max=7.0))
    allowed_pos: list[str] = Field(default_factory=lambda: ["NOUN", "ADJ", "VERB"])


class VocabularyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "en"
    game: GameVocabularySideConfig = Field(default_factory=GameVocabularySideConfig)
    clue: ClueVocabularySideConfig = Field(default_factory=ClueVocabularySideConfig)


class GuesserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extra_candidates: int = Field(default=3, ge=0, le=50)
    sampling_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Softmax temperature for stochastic guess picks (0 disables sampling).",
    )
    sampling_top_k: int = Field(
        default=0,
        ge=0,
        le=25,
        description="Sample only from top-K ranked candidates (0 means all unrevealed cards).",
    )


class ScoringConfig(BaseModel):
    """Scoring and EV/rerank tuning knobs for the spymaster."""

    model_config = ConfigDict(extra="forbid")

    llm_rerank: bool = True
    blend_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

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
    ambition_weight: float | None = Field(default=None, ge=0.0, le=2.0)
    margin_weight: float | None = Field(default=None, ge=0.0, le=2.0)
    freq_weight: float | None = Field(default=None, ge=0.0, le=2.0)
    assassin_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    opponent_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    undercluster_penalty_weight: float | None = Field(default=None, ge=0.0, le=2.0)
    margin_floor: float | None = Field(default=None, ge=-1.0, le=1.0)
    assassin_ceiling: float | None = Field(default=None, ge=-1.0, le=1.0)
    mc_temperature: float | None = Field(default=None, ge=0.01, le=5.0)
    mc_rank_bias: float | None = Field(default=None, ge=0.0, le=10.0)
    reward_friendly: float | None = Field(default=None, ge=-10.0, le=10.0)
    reward_neutral: float | None = Field(default=None, ge=-10.0, le=10.0)
    reward_opponent: float | None = Field(default=None, ge=-10.0, le=10.0)
    reward_assassin: float | None = Field(default=None, ge=-20.0, le=0.0)
    mc_trials: int = Field(
        default=96,
        ge=1,
        le=2000,
        description="Monte Carlo rollouts per `(clue, N)` scoring candidate.",
    )
    adaptive_mc_base_trials: int = Field(default=64, ge=1, le=2000)
    adaptive_mc_extra_trials: int = Field(default=96, ge=0, le=5000)
    adaptive_mc_ev_band: float = Field(default=0.10, ge=0.0, le=5.0)

    lane_target_fractions: list[float] = Field(
        default_factory=lambda: [0.18, 0.42, 0.22, 0.10, 0.05, 0.02, 0.01],
        description="Target fractions for N=1..7 shortlist lanes.",
    )
    lane_quality_delta_ev: float = Field(default=0.20, ge=0.0, le=5.0)
    lane_max_n: int = Field(default=7, ge=2, le=9)

    ev_llm_gain: float = Field(default=0.35, ge=0.0, le=2.0)
    ev_llm_temperature: float = Field(default=0.20, gt=0.0, le=5.0)


class EvalAgentConfigFile(BaseModel):
    """YAML schema for `codenames-ai eval --config file.yaml`."""

    model_config = ConfigDict(extra="forbid")

    label: str = "default"
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)
    exclusions_path: Path | None = None

    top_k_trace: int = Field(
        default=200,
        ge=1,
        le=10_000,
        description="Max candidates kept in SpymasterTrace after rerank.",
    )
    embedding_top_k: int = Field(
        default=20,
        ge=1,
        le=10_000,
        description="Top embedding-scored candidates sent to the spymaster LLM.",
    )
    guesser: GuesserConfig = Field(default_factory=GuesserConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    @model_validator(mode="after")
    def _validate_lane_fractions(self) -> "EvalAgentConfigFile":
        if len(self.scoring.lane_target_fractions) != 7:
            raise ValueError("lane_target_fractions must provide exactly 7 values (N=1..7)")
        if sum(self.scoring.lane_target_fractions) <= 0.0:
            raise ValueError("lane_target_fractions must sum to a positive value")
        return self


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
