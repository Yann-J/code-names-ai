from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from codenames_ai.agent.interfaces import NoLegalClueError, Spymaster
from codenames_ai.agent.rerank import SpymasterReranker
from codenames_ai.agent.scoring import ScoringWeights, freq_bonus
from codenames_ai.agent.trace import Candidate, ScoreComponents, SpymasterTrace
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Card, Color, SpymasterView
from codenames_ai.game.rules import RuleStrictness, is_legal_clue
from codenames_ai.vocab.models import Vocabulary

logger = logging.getLogger(__name__)

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class _ClueIndex:
    """Pre-computed alignment of clue vocabulary into the embedding matrix."""

    surfaces: list[str]
    lemmas: list[str]
    zipfs: np.ndarray  # (M,)
    matrix_indices: np.ndarray  # (M,) int

    @classmethod
    def build(cls, vocab: Vocabulary, matrix: EmbeddingMatrix) -> "_ClueIndex":
        df = vocab.df
        keep_surfaces: list[str] = []
        keep_lemmas: list[str] = []
        keep_zipfs: list[float] = []
        keep_idx: list[int] = []
        for surface, lemma, zipf in zip(
            df["surface"].tolist(),
            df["lemma"].tolist(),
            df["zipf"].tolist(),
            strict=True,
        ):
            idx = matrix.index_of(surface)
            if idx is None:
                continue
            keep_surfaces.append(surface)
            keep_lemmas.append(lemma)
            keep_zipfs.append(float(zipf))
            keep_idx.append(idx)
        return cls(
            surfaces=keep_surfaces,
            lemmas=keep_lemmas,
            zipfs=np.array(keep_zipfs, dtype=np.float32),
            matrix_indices=np.array(keep_idx, dtype=np.int64),
        )


class AISpymaster(Spymaster):
    """Embedding-cluster spymaster (M3, no LLM yet).

    Algorithm B:
      1. Compute clue×board similarity (one matmul against the precomputed clue
         index).
      2. For each clue, sort friendly similarities descending; the only subsets
         worth scoring are prefixes (size 1..F).
      3. Apply hard vetoes (margin floor, assassin ceiling) and the legality
         rule. Score survivors via `ScoringWeights`.
      4. Pick the highest-scoring candidate; keep the top `top_k` for the trace.
    """

    def __init__(
        self,
        matrix: EmbeddingMatrix,
        clue_vocabulary: Vocabulary,
        *,
        risk: float = 0.5,
        weights: ScoringWeights | None = None,
        rule_strictness: RuleStrictness = "lemma_substring",
        top_k: int = 50,
        reranker: SpymasterReranker | None = None,
    ) -> None:
        self.matrix = matrix
        self.clue_vocabulary = clue_vocabulary
        self.weights = weights or ScoringWeights.from_risk(risk)
        self.rule_strictness = rule_strictness
        self.top_k = top_k
        self.reranker = reranker
        self._clue_index = _ClueIndex.build(clue_vocabulary, matrix)
        if not self._clue_index.surfaces:
            raise ValueError(
                "no overlap between clue vocabulary and embedding matrix surfaces"
            )

    def give_clue(self, view: SpymasterView) -> SpymasterTrace:
        active = list(view.board.active())
        self._validate_board(active)

        team = view.team
        opp = team.opponent()

        friendly = [c for c in active if c.color == team]
        opponent = [c for c in active if c.color == opp]
        neutral = [c for c in active if c.color == Color.NEUTRAL]
        assassin = next((c for c in active if c.color == Color.ASSASSIN), None)

        if not friendly:
            raise ValueError("no friendly cards remaining; nothing to clue toward")

        clue_idx = self._clue_index
        clue_vecs = self.matrix.vectors[clue_idx.matrix_indices]  # (M, dim)

        sim_friendly = clue_vecs @ self._stack(friendly).T  # (M, F)
        sim_opponent = (
            clue_vecs @ self._stack(opponent).T
            if opponent
            else np.full((len(clue_vecs), 0), 0.0, dtype=np.float32)
        )
        sim_neutral = (
            clue_vecs @ self._stack(neutral).T
            if neutral
            else np.full((len(clue_vecs), 0), 0.0, dtype=np.float32)
        )
        sim_assassin = (
            clue_vecs @ self.matrix[assassin.word]
            if assassin is not None
            else np.full(len(clue_vecs), _NEG_INF, dtype=np.float32)
        )

        # Per-row sorted friendly similarities + their card identities.
        sort_idx = np.argsort(-sim_friendly, axis=1)
        sorted_friendly = np.take_along_axis(sim_friendly, sort_idx, axis=1)

        best_opp = (
            sim_opponent.max(axis=1)
            if sim_opponent.shape[1] > 0
            else np.full(len(clue_vecs), _NEG_INF, dtype=np.float32)
        )
        best_neutral = (
            sim_neutral.max(axis=1)
            if sim_neutral.shape[1] > 0
            else np.full(len(clue_vecs), _NEG_INF, dtype=np.float32)
        )
        best_non_friendly = np.maximum.reduce([best_opp, best_neutral, sim_assassin])

        candidates, veto_count, illegal_count = self._score_all(
            clue_idx=clue_idx,
            friendly=friendly,
            sort_idx=sort_idx,
            sorted_friendly=sorted_friendly,
            best_opp=best_opp,
            best_non_friendly=best_non_friendly,
            sim_assassin=sim_assassin,
            active_cards=active,
        )

        candidates.sort(key=lambda c: c.score, reverse=True)

        if self.reranker is not None and candidates:
            # Per PRD: only the top-K shortlist competes after rerank. The
            # tail is discarded — embedding-only scores are unbounded while
            # blended scores live in [0, 1], so they aren't comparable, and
            # a non-shortlisted candidate "winning" by embedding score alone
            # would defeat the purpose of asking the LLM.
            shortlist = candidates[: self.reranker.top_k]
            candidates = list(self.reranker.rerank(shortlist, view))
            candidates.sort(key=lambda c: c.score, reverse=True)

        top = tuple(candidates[: self.top_k])

        if not top:
            raise NoLegalClueError(
                f"no candidates passed vetoes "
                f"(rejected: {veto_count} by margin/assassin, {illegal_count} by legality)"
            )

        chosen = top[0]
        logger.info(
            "spymaster chose %r (n=%d, score=%.3f, margin=%.3f)",
            chosen.clue,
            chosen.n,
            chosen.score,
            chosen.margin,
        )
        return SpymasterTrace(
            chosen=chosen,
            top_candidates=top,
            weights=self.weights,
            veto_count=veto_count,
            illegal_count=illegal_count,
        )

    def _validate_board(self, active: list[Card]) -> None:
        missing = [c.word for c in active if c.word not in self.matrix]
        if missing:
            raise ValueError(
                f"board words missing from embedding matrix: {missing}. "
                f"Build the matrix from a vocabulary that covers all board words."
            )

    def _stack(self, cards: list[Card]) -> np.ndarray:
        if not cards:
            return np.zeros((0, self.matrix.dim), dtype=np.float32)
        return self.matrix.vectors[
            np.array([self.matrix.surface_to_index[c.word] for c in cards])
        ]

    def _score_all(
        self,
        *,
        clue_idx: _ClueIndex,
        friendly: list[Card],
        sort_idx: np.ndarray,
        sorted_friendly: np.ndarray,
        best_opp: np.ndarray,
        best_non_friendly: np.ndarray,
        sim_assassin: np.ndarray,
        active_cards: list[Card],
    ) -> tuple[list[Candidate], int, int]:
        weights = self.weights
        F = sorted_friendly.shape[1]
        candidates: list[Candidate] = []
        veto_count = 0
        illegal_count = 0

        for i in range(len(clue_idx.surfaces)):
            surface = clue_idx.surfaces[i]
            lemma = clue_idx.lemmas[i]
            zipf = float(clue_idx.zipfs[i])

            assassin_sim = float(sim_assassin[i])
            opp_sim = float(best_opp[i]) if best_opp[i] != _NEG_INF else 0.0
            row_sorted = sorted_friendly[i]

            # Legality (filters surface and lemma vs all active cards).
            if not is_legal_clue(
                clue_surface=surface,
                clue_lemma=lemma,
                active_cards=active_cards,
                strictness=self.rule_strictness,
            ):
                illegal_count += F  # one illegal clue → all N evaluations rejected
                continue

            for n in range(1, F + 1):
                friendly_min_sim = float(row_sorted[n - 1])
                margin = friendly_min_sim - float(best_non_friendly[i])

                if margin < weights.margin_floor:
                    veto_count += 1
                    continue
                if assassin_sim > weights.assassin_ceiling:
                    veto_count += 1
                    continue

                ambition_bonus = weights.ambition_weight * (n - 1)
                margin_bonus = weights.margin_weight * margin
                fb = freq_bonus(zipf, weights.freq_weight)
                assassin_penalty = weights.assassin_weight * max(assassin_sim, 0.0)
                opponent_penalty = weights.opponent_weight * max(opp_sim, 0.0)

                components = ScoreComponents(
                    friendly_min_sim=friendly_min_sim,
                    ambition_bonus=ambition_bonus,
                    margin_bonus=margin_bonus,
                    freq_bonus=fb,
                    assassin_penalty=assassin_penalty,
                    opponent_penalty=opponent_penalty,
                )
                target_indices = sort_idx[i, :n]
                targets = tuple(friendly[int(idx)].word for idx in target_indices)
                emb_score = components.total
                candidates.append(
                    Candidate(
                        clue=surface,
                        targets=targets,
                        n=n,
                        score=emb_score,
                        embedding_score=emb_score,
                        components=components,
                        margin=margin,
                        zipf=zipf,
                    )
                )

        return candidates, veto_count, illegal_count
