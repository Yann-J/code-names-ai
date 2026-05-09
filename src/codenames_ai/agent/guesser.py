from __future__ import annotations

import logging

import numpy as np

from codenames_ai.agent.interfaces import Guesser
from codenames_ai.agent.rerank import GuesserReranker
from codenames_ai.agent.scoring import StopPolicy
from codenames_ai.agent.trace import CandidateGuess, GuesserTrace
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Card, Clue, GuesserView

logger = logging.getLogger(__name__)


class AIGuesser(Guesser):
    """Embedding-only guesser (M4, no LLM yet).

    Algorithm:
      1. Score every unrevealed card by cosine similarity to the clue word.
      2. Sort descending; always commit pick #1.
      3. Picks 2..N: commit while similarity ≥ `confidence_floor`.
      4. Bonus N+1 pick: commit only when the gap between Nth and (N+1)th
         is below `bonus_gap_threshold`
         (negative threshold disables the bonus).
    """

    def __init__(
        self,
        matrix: EmbeddingMatrix,
        *,
        risk: float = 0.5,
        stop_policy: StopPolicy | None = None,
        reranker: GuesserReranker | None = None,
        sampling_temperature: float = 0.0,
        sampling_top_k: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.matrix = matrix
        self.stop_policy = stop_policy or StopPolicy.from_risk(risk)
        self.reranker = reranker
        self.sampling_temperature = max(0.0, float(sampling_temperature))
        self.sampling_top_k = max(0, int(sampling_top_k))
        self.rng = rng or np.random.default_rng()

    def guess(self, view: GuesserView, clue: Clue) -> GuesserTrace:
        active = list(view.board.active())
        self._validate(active, clue)

        if clue.is_pass():
            return GuesserTrace(
                candidates=(),
                guesses=(),
                stop_policy=self.stop_policy,
                bonus_attempted=False,
                stop_reason="pass_clue",
            )

        clue_vec = self.matrix[clue.word]
        sims = self._sims_for(active, clue_vec)
        order = np.argsort(-sims)

        ranked = [
            CandidateGuess(
                word=active[i].word,
                similarity=float(sims[i]),
                score=float(sims[i]),
                rank=rank,
                committed=False,
                is_bonus=False,
            )
            for rank, i in enumerate(order)
        ]

        logger.debug(
            "guesser embedding ranking: clue=%r count=%s (%d unrevealed cards), "
            "cosine sim descending:",
            clue.word,
            clue.count,
            len(active),
        )
        for cand in ranked:
            logger.debug(
                "  emb rank %d %r cosine=%.6f",
                cand.rank,
                cand.word,
                cand.similarity,
            )

        if self.reranker is not None:
            ranked = list(self.reranker.rerank(ranked, view, clue))
            ranked.sort(key=lambda c: c.score, reverse=True)
            # Re-stamp ranks after re-sorting.
            ranked = [
                CandidateGuess(
                    word=c.word,
                    similarity=c.similarity,
                    score=c.score,
                    rank=new_rank,
                    committed=False,
                    is_bonus=False,
                    llm_score=c.llm_score,
                    llm_reason=c.llm_reason,
                )
                for new_rank, c in enumerate(ranked)
            ]
            logger.info(
                "guesser ranking after rerank: %d cards by blended score:",
                len(ranked),
            )
            for cand in ranked:
                llm = f"{cand.llm_score:.4f}" if cand.llm_score is not None else "—"
                reason = (cand.llm_reason or "").strip()
                tail = f" | {reason}" if reason else ""
                logger.info(
                    "  rank %d %r blend=%.6f cosine=%.6f llm=%s%s",
                    cand.rank,
                    cand.word,
                    cand.score,
                    cand.similarity,
                    llm,
                    tail,
                )

        guesses, bonus_attempted, stop_reason = self._apply_stop_policy(
            ranked, clue.count
        )
        return GuesserTrace(
            candidates=tuple(ranked),
            guesses=tuple(g.word for g in guesses),
            stop_policy=self.stop_policy,
            bonus_attempted=bonus_attempted,
            stop_reason=stop_reason,
        )

    def _validate(self, active: list[Card], clue: Clue) -> None:
        if clue.is_pass():
            return
        if clue.word not in self.matrix:
            raise ValueError(
                f"clue word {clue.word!r} not in embedding matrix; "
                f"build a matrix that covers anticipated clue words."
            )
        missing = [c.word for c in active if c.word not in self.matrix]
        if missing:
            raise ValueError(f"unrevealed board words missing from matrix: {missing}")

    def _sims_for(self, cards: list[Card], clue_vec: np.ndarray) -> np.ndarray:
        if not cards:
            return np.zeros(0, dtype=np.float32)
        idx = np.array([self.matrix.surface_to_index[c.word] for c in cards])
        return self.matrix.vectors[idx] @ clue_vec

    def _apply_stop_policy(
        self,
        ranked: list[CandidateGuess],
        n: int,
    ) -> tuple[list[CandidateGuess], bool, str]:
        if not ranked:
            return [], False, "no_more_candidates"

        policy = self.stop_policy
        guesses: list[CandidateGuess] = []
        # Always commit to one pick (declining outright is a wasted turn).
        first_rank = self._next_rank(ranked)
        guesses.append(self._commit(ranked, first_rank, is_bonus=False))

        # Picks 2..N — gate on the (potentially blended) `score`, not the raw
        # cosine similarity, so an LLM rerank can lower a card's effective
        # confidence and trigger an early stop.
        for _ in range(1, min(n, len(ranked))):
            next_rank = self._next_rank(ranked)
            cand = ranked[next_rank]
            if cand.score < policy.confidence_floor:
                return guesses, False, "confidence_floor"
            guesses.append(self._commit(ranked, next_rank, is_bonus=False))

        if len(guesses) < n:
            return guesses, False, "no_more_candidates"

        # Bonus N+1 attempt
        if n < len(ranked):
            nth_score = guesses[-1].score
            next_rank = self._next_rank(ranked)
            bonus_cand = ranked[next_rank]
            gap = nth_score - bonus_cand.score
            if gap < policy.bonus_gap_threshold:
                guesses.append(self._commit(ranked, next_rank, is_bonus=True))
                return guesses, True, "reached_n_plus_bonus"

        return guesses, False, "reached_n"

    def _next_rank(self, ranked: list[CandidateGuess]) -> int:
        available = [c for c in ranked if not c.committed]
        if not available:
            raise ValueError("no available candidates")

        if self.sampling_temperature <= 0.0:
            return available[0].rank

        top_k = self.sampling_top_k if self.sampling_top_k > 0 else len(available)
        pool = available[:top_k]
        scores = np.array([c.score for c in pool], dtype=np.float64)
        logits = scores / self.sampling_temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            return pool[0].rank
        probs /= probs_sum
        draw = self.rng.random()
        cdf = np.cumsum(probs)
        chosen_idx = int(np.searchsorted(cdf, draw, side="right"))
        chosen_idx = min(chosen_idx, len(pool) - 1)
        return pool[chosen_idx].rank

    @staticmethod
    def _commit(
        ranked: list[CandidateGuess], rank: int, *, is_bonus: bool
    ) -> CandidateGuess:
        original = ranked[rank]
        committed = CandidateGuess(
            word=original.word,
            similarity=original.similarity,
            score=original.score,
            rank=original.rank,
            committed=True,
            is_bonus=is_bonus,
            llm_score=original.llm_score,
            llm_reason=original.llm_reason,
        )
        ranked[rank] = committed
        return committed
