from __future__ import annotations

from dataclasses import dataclass, replace

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.interfaces import Guesser, Spymaster
from codenames_ai.agent.llm_guess_policy import ContinueGate
from codenames_ai.agent.llm_guess_scorer import LLMGuessScorer, ScorerConfig
from codenames_ai.agent.llm_guesser import LLMGuesser
from codenames_ai.agent.rerank import GuesserReranker, SpymasterReranker
from codenames_ai.agent.risk_context import (
    DynamicRiskAIGuesser,
    DynamicRiskAISpymaster,
    DynamicRiskPolicy,
)
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.cli.eval_config import EvalAgentConfigFile
from codenames_ai.config import Config
from codenames_ai.embedding.builder import load_or_build_embedding_matrix
from codenames_ai.embedding.download import fasttext_default_filename
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.embedding.provider import FastTextProvider
from codenames_ai.llm.cache import LLMCache
from codenames_ai.llm.provider import OpenAICompatibleProvider
from codenames_ai.storage import StoragePaths
from codenames_ai.vocab.builder import load_or_build_vocabulary
from codenames_ai.vocab.filters import load_exclusions
from codenames_ai.vocab.models import Vocabulary, VocabConfig


@dataclass(frozen=True)
class EvalRuntime:
    game_vocab: Vocabulary
    clue_vocab: Vocabulary
    matrix: EmbeddingMatrix
    clue_surface_exclusions: frozenset[str]
    spymaster: Spymaster
    guesser: Guesser
    dynamic_risk_policy: DynamicRiskPolicy


def _vocab_config(
    game: EvalAgentConfigFile, *, game_words: bool
) -> VocabConfig:
    vocab = game.vocabulary
    if game_words:
        return VocabConfig(
            language=vocab.language,
            zipf_min=vocab.game.zipf.min,
            zipf_max=vocab.game.zipf.max,
            allowed_pos=frozenset(vocab.game.allowed_pos),
            exclusions_path=vocab.exclusions_path,
        )
    return VocabConfig(
        language=vocab.language,
        zipf_min=vocab.clue.zipf.min,
        zipf_max=vocab.clue.zipf.max,
        allowed_pos=frozenset(vocab.clue.allowed_pos),
        exclusions_path=vocab.exclusions_path,
    )


def _dynamic_risk_policy(cfg: EvalAgentConfigFile) -> DynamicRiskPolicy:
    d = cfg.dynamic_risk
    return DynamicRiskPolicy(
        enabled=bool(d.enabled),
        s=float(d.s),
        min_risk=float(d.min_risk),
        max_risk=float(d.max_risk),
        beta_margin_floor=float(d.beta_margin_floor),
        beta_assassin_ceiling=float(d.beta_assassin_ceiling),
        beta_confidence_floor=float(d.beta_confidence_floor),
        beta_bonus_gap=float(d.beta_bonus_gap),
    )


def build_eval_runtime(cfg: EvalAgentConfigFile, app: Config) -> EvalRuntime:
    """Load vocab, matrix, and AI players for eval runs."""
    scoring = cfg.scoring
    storage = StoragePaths.from_config(app)
    storage.ensure()

    game_vocab = load_or_build_vocabulary(
        _vocab_config(cfg, game_words=True), storage
    )
    clue_vocab = load_or_build_vocabulary(
        _vocab_config(cfg, game_words=False), storage
    )

    model_path = app.fasttext_path or (
        storage.models_dir / fasttext_default_filename(cfg.vocabulary.language)
    )
    provider = FastTextProvider(model_path)
    matrix = load_or_build_embedding_matrix(clue_vocab, provider, storage)

    guesser_mode = cfg.guesser.mode
    needs_llm = scoring.llm_rerank or guesser_mode == "llm_primary"
    llm: OpenAICompatibleProvider | None = None
    if needs_llm:
        model = app.llm_model or "gpt-4o-mini"
        base_url = app.llm_api or "https://api.openai.com/v1"
        key = app.llm_key.get_secret_value() if app.llm_key else ""
        if not key:
            raise RuntimeError(
                "An LLM-backed feature is enabled (scoring.llm_rerank or "
                "guesser.mode=llm_primary) but LLM_KEY (or "
                "CODENAMES_AI_LLM_KEY) is not set. Use embedding-only eval "
                "with --embedding-only or set credentials."
            )
        cache = LLMCache(storage.llm_cache_path)
        llm = OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=key,
            cache=cache,
            temperature=0.0,
        )

    spy_reranker = None
    guess_reranker = None
    if scoring.llm_rerank and llm is not None:
        spy_reranker = SpymasterReranker(
            llm,
            top_k=scoring.embedding_top_k,
            blend_alpha=scoring.blend_alpha,
        )
        if guesser_mode == "embedding":
            guess_reranker = GuesserReranker(
                llm,
                extra_candidates=cfg.guesser.extra_candidates,
                blend_alpha=scoring.blend_alpha,
            )

    base = ScoringWeights.from_risk(cfg.risk.base_risk)
    updates: dict[str, object] = {
        "mc_trials": scoring.mc_trials,
        "adaptive_mc_base_trials": scoring.adaptive_mc_base_trials,
        "adaptive_mc_extra_trials": scoring.adaptive_mc_extra_trials,
        "adaptive_mc_ev_band": scoring.adaptive_mc_ev_band,
        "lane_max_n": scoring.lane_max_n,
    }
    optional_overrides = {
        "margin_floor": scoring.margin_floor,
        "assassin_ceiling": scoring.assassin_ceiling,
        "mc_temperature": scoring.mc_temperature,
        "mc_rank_bias": scoring.mc_rank_bias,
        "reward_friendly": scoring.reward_friendly,
        "reward_neutral": scoring.reward_neutral,
        "reward_opponent": scoring.reward_opponent,
        "reward_assassin": scoring.reward_assassin,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            updates[key] = value
    spy_weights = replace(base, **updates)
    clue_surface_exclusions = load_exclusions(cfg.vocabulary.exclusions_path)
    inner_spy = AISpymaster(
        matrix,
        clue_vocab,
        risk=cfg.risk.base_risk,
        top_k=cfg.top_k_trace,
        reranker=spy_reranker,
        weights=spy_weights,
        clue_surface_exclusions=clue_surface_exclusions,
    )
    dyn = _dynamic_risk_policy(cfg)
    spymaster = (
        DynamicRiskAISpymaster(inner_spy, base_risk=cfg.risk.base_risk, policy=dyn)
        if dyn.enabled
        else inner_spy
    )
    if guesser_mode == "llm_primary":
        if llm is None:
            raise RuntimeError(
                "guesser.mode=llm_primary requires LLM credentials (LLM_KEY)."
            )
        llm_cfg = cfg.guesser.llm
        scorer = LLMGuessScorer(
            llm,
            config=ScorerConfig(
                schema_mode=llm_cfg.schema_mode,
                retry_count=llm_cfg.retry_count,
                keep_raw_response=llm_cfg.keep_raw_response,
            ),
        )
        gate = ContinueGate(
            min_combined=llm_cfg.min_combined,
            min_margin_to_second=llm_cfg.min_margin_to_second,
            min_fit=llm_cfg.min_fit,
        )
        guesser = LLMGuesser(
            scorer,
            lambda_danger=llm_cfg.lambda_danger,
            gate=gate,
            embedding_matrix=matrix if llm_cfg.embedding_fallback else None,
        )
        # LLM-primary policy owns its own continue/stop semantics — bypass the
        # embedding-era dynamic-risk wrapper which only modulates StopPolicy.
    else:
        inner_g = AIGuesser(
            matrix,
            risk=cfg.risk.base_risk,
            reranker=guess_reranker,
            sampling_temperature=cfg.guesser.sampling_temperature,
            sampling_top_k=cfg.guesser.sampling_top_k,
        )
        guesser = (
            DynamicRiskAIGuesser(inner_g, base_risk=cfg.risk.base_risk, policy=dyn)
            if dyn.enabled
            else inner_g
        )
    return EvalRuntime(
        game_vocab=game_vocab,
        clue_vocab=clue_vocab,
        matrix=matrix,
        clue_surface_exclusions=clue_surface_exclusions,
        spymaster=spymaster,  # may be DynamicRiskAISpymaster when enabled
        guesser=guesser,
        dynamic_risk_policy=dyn,
    )
