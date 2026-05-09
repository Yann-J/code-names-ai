from __future__ import annotations

from pathlib import Path

import yaml
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskConfig(BaseModel):
    """Knobs that tune how aggressively the AI plays (weights, stopping, vetoes)."""

    model_config = ConfigDict(extra="forbid")

    base_risk: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maps to spymaster scoring vetoes / MC softmax and guesser stop policy.",
    )


class DynamicRiskConfig(BaseModel):
    """Optional board-aware adjustment of ``base_risk`` and selected veto/stop scalars."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    s: float = Field(
        default=0.12,
        ge=0.0,
        le=2.0,
        description="Strength of Δ on effective_risk via exp(−s·Δ).",
    )
    min_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    max_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    beta_margin_floor: float = Field(default=0.25, ge=0.0, le=3.0)
    beta_assassin_ceiling: float = Field(default=0.25, ge=0.0, le=3.0)
    beta_confidence_floor: float = Field(default=0.25, ge=0.0, le=3.0)
    beta_bonus_gap: float = Field(default=0.25, ge=0.0, le=3.0)

    @model_validator(mode="after")
    def _min_le_max(self) -> "DynamicRiskConfig":
        if float(self.min_risk) > float(self.max_risk):
            raise ValueError("dynamic_risk.min_risk must be <= max_risk")
        return self


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
    exclusions_path: Path | None = Field(
        default=None,
        description=(
            "Optional text file of words to exclude from vocab building (one per line). "
            "Relative paths resolve against the directory of the YAML file passed to the CLI."
        ),
    )
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
    """Monte Carlo EV scoring, hard vetoes, and LLM rerank blend."""

    model_config = ConfigDict(extra="forbid")

    llm_rerank: bool = True
    blend_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Final spymaster score: alpha*EV + (1-alpha)*(LLM_confidence*N_eff).",
    )

    margin_floor: float | None = Field(default=None, ge=-1.0, le=1.0)
    assassin_ceiling: float | None = Field(default=None, ge=-1.0, le=1.0)
    mc_temperature: float | None = Field(default=None, ge=0.01, le=5.0)
    mc_rank_bias: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description=(
            "MC softmax: penalize logits by bias*log1p(rank-by-sim). "
            "0 = cosine-only stochastic; higher = greedier stochastic."
        ),
    )
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

    lane_max_n: int = Field(
        default=7,
        ge=1,
        le=9,
        description="Max target count ``N`` to evaluate per clue (prefix length cap).",
    )
    embedding_top_k: int = Field(
        default=20,
        ge=1,
        le=10_000,
        description=(
            "After embedding / EV scoring: how many top candidates are sent to the "
            "spymaster LLM reranker when ``llm_rerank`` is enabled."
        ),
    )


class EvalAgentConfigFile(BaseModel):
    """YAML schema for `codenames-ai eval --config file.yaml`."""

    model_config = ConfigDict(extra="forbid")

    label: str = "default"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    dynamic_risk: DynamicRiskConfig = Field(default_factory=DynamicRiskConfig)
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)

    top_k_trace: int = Field(
        default=200,
        ge=1,
        le=10_000,
        description="Max candidates kept in SpymasterTrace after rerank.",
    )
    guesser: GuesserConfig = Field(default_factory=GuesserConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    @model_validator(mode="before")
    @classmethod
    def _legacy_flat_risk(cls, data: Any) -> Any:
        """Accept legacy top-level ``risk: 0.5`` as ``risk: {base_risk: 0.5}``."""
        if not isinstance(data, dict):
            return data
        r = data.get("risk")
        if isinstance(r, (int, float)):
            merged = {k: v for k, v in data.items() if k != "risk"}
            merged["risk"] = {"base_risk": float(r)}
            return merged
        return data


def _deep_merge_mapping(base: dict, override: dict) -> dict:
    """Recursively merge mappings; ``override`` wins. Lists and scalars replace."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_mapping(out[key], value)
        else:
            out[key] = value
    return out


def _extends_to_paths(raw: object, *, relative_to: Path) -> list[Path]:
    if isinstance(raw, str):
        seq = [raw]
    elif isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        seq = list(raw)
    else:
        raise ValueError("extends must be a string or a list of strings")
    resolved: list[Path] = []
    for item in seq:
        p = Path(item)
        resolved.append(p.resolve() if p.is_absolute() else (relative_to.parent / p).resolve())
    return resolved


def load_eval_yaml_merged(path: Path, *, _chain: frozenset[Path] | None = None) -> dict:
    """Load YAML, applying ``extends`` (inherit) depth-first; child overrides parent."""
    path = path.resolve()
    chain = _chain or frozenset()
    if path in chain:
        msg = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"cyclic config extends: {msg}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    inherited = data.pop("extends", None)
    merged: dict = {}
    if inherited is not None:
        for parent in _extends_to_paths(inherited, relative_to=path):
            merged = _deep_merge_mapping(merged, load_eval_yaml_merged(parent, _chain=chain | {path}))
    return _deep_merge_mapping(merged, data)


def load_eval_yaml(path: Path) -> tuple[EvalAgentConfigFile, str]:
    """Load and validate a YAML file; return `(config, config_hash_hex16)`.

    If the file defines ``extends``, parent file(s) are merged first (each path is
    relative to the file that references it). Later keys override earlier ones; the
    loaded file wins over all parents. The digest covers the merged configuration.
    """
    import hashlib
    import json

    path = path.resolve()
    data = load_eval_yaml_merged(path)
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    cfg = EvalAgentConfigFile.model_validate(data)
    vocab = cfg.vocabulary
    if vocab.exclusions_path is not None and not vocab.exclusions_path.is_absolute():
        rel = path.parent / vocab.exclusions_path
        cfg = cfg.model_copy(
            update={
                "vocabulary": vocab.model_copy(update={"exclusions_path": rel.resolve()})
            }
        )
    return cfg, digest
