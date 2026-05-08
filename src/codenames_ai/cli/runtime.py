from __future__ import annotations

from dataclasses import dataclass, replace

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.rerank import GuesserReranker, SpymasterReranker
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
from codenames_ai.vocab.models import Vocabulary, VocabConfig


@dataclass(frozen=True)
class EvalRuntime:
    game_vocab: Vocabulary
    clue_vocab: Vocabulary
    matrix: EmbeddingMatrix
    spymaster: AISpymaster
    guesser: AIGuesser


def _vocab_config(game: EvalAgentConfigFile, *, game_words: bool) -> VocabConfig:
    vocab = game.vocabulary
    if game_words:
        return VocabConfig(
            language=vocab.language,
            zipf_min=vocab.game.zipf.min,
            zipf_max=vocab.game.zipf.max,
            allowed_pos=frozenset(vocab.game.allowed_pos),
            exclusions_path=game.exclusions_path,
        )
    return VocabConfig(
        language=vocab.language,
        zipf_min=vocab.clue.zipf.min,
        zipf_max=vocab.clue.zipf.max,
        allowed_pos=frozenset(vocab.clue.allowed_pos),
        exclusions_path=game.exclusions_path,
    )


def build_eval_runtime(cfg: EvalAgentConfigFile, app: Config) -> EvalRuntime:
    scoring = cfg.scoring
    """Load vocab, matrix, and AI players for eval runs."""
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

    spy_reranker = None
    guess_reranker = None
    if scoring.llm_rerank:
        model = app.llm_model or "gpt-4o-mini"
        base_url = app.llm_api or "https://api.openai.com/v1"
        key = app.llm_key.get_secret_value() if app.llm_key else ""
        if not key:
            raise RuntimeError(
                "LLM rerank is enabled but LLM_KEY "
                "(or CODENAMES_AI_LLM_KEY) is not set. "
                "Use embedding-only eval with --embedding-only "
                "or set credentials."
            )
        cache = LLMCache(storage.llm_cache_path)
        llm = OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=key,
            cache=cache,
            temperature=0.0,
        )
        spy_reranker = SpymasterReranker(
            llm,
            top_k=cfg.embedding_top_k,
            blend_alpha=scoring.blend_alpha,
            ev_llm_gain=scoring.ev_llm_gain,
            ev_llm_temperature=scoring.ev_llm_temperature,
        )
        guess_reranker = GuesserReranker(
            llm,
            extra_candidates=cfg.guesser.extra_candidates,
            blend_alpha=scoring.blend_alpha,
        )

    base = ScoringWeights.from_risk(cfg.risk)
    updates: dict[str, object] = {
        "prefer_min_targets": scoring.prefer_min_targets,
        "expected_reward_weight": scoring.expected_reward_weight,
        "mc_trials": scoring.mc_trials,
        "adaptive_mc_base_trials": scoring.adaptive_mc_base_trials,
        "adaptive_mc_extra_trials": scoring.adaptive_mc_extra_trials,
        "adaptive_mc_ev_band": scoring.adaptive_mc_ev_band,
        "lane_target_fractions": tuple(scoring.lane_target_fractions),
        "lane_quality_delta_ev": scoring.lane_quality_delta_ev,
        "lane_max_n": scoring.lane_max_n,
    }
    optional_overrides = {
        "ambition_weight": scoring.ambition_weight,
        "margin_weight": scoring.margin_weight,
        "freq_weight": scoring.freq_weight,
        "assassin_weight": scoring.assassin_weight,
        "opponent_weight": scoring.opponent_weight,
        "undercluster_penalty_weight": scoring.undercluster_penalty_weight,
        "margin_floor": scoring.margin_floor,
        "assassin_ceiling": scoring.assassin_ceiling,
        "mc_temperature": scoring.mc_temperature,
        "reward_friendly": scoring.reward_friendly,
        "reward_neutral": scoring.reward_neutral,
        "reward_opponent": scoring.reward_opponent,
        "reward_assassin": scoring.reward_assassin,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            updates[key] = value
    spy_weights = replace(base, **updates)
    spymaster = AISpymaster(
        matrix,
        clue_vocab,
        risk=cfg.risk,
        top_k=cfg.top_k_trace,
        reranker=spy_reranker,
        weights=spy_weights,
    )
    guesser = AIGuesser(matrix, risk=cfg.risk, reranker=guess_reranker)
    return EvalRuntime(
        game_vocab=game_vocab,
        clue_vocab=clue_vocab,
        matrix=matrix,
        spymaster=spymaster,
        guesser=guesser,
    )
