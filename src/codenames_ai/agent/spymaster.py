from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from codenames_ai.agent.interfaces import NoLegalClueError, Spymaster
from codenames_ai.agent.rerank import SpymasterReranker
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.trace import Candidate, ScoreComponents, SpymasterTrace
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Card, Color, SpymasterView
from codenames_ai.game.rules import RuleStrictness, is_legal_clue
from codenames_ai.vocab.models import Vocabulary

logger = logging.getLogger(__name__)

_NEG_INF = float("-inf")


def _dedupe_candidates_by_clue(
    ordered: list[Candidate], *, limit: int
) -> tuple[Candidate, ...]:
    """Keep at most one row per clue surface (best score first).

    Scoring emits one candidate per (clue, n); without deduping, ``top_candidates``
    repeats the same clue word with different N/target subsets.
    """

    seen: set[str] = set()
    out: list[Candidate] = []
    for c in ordered:
        if c.clue in seen:
            continue
        seen.add(c.clue)
        out.append(c)
        if len(out) >= limit:
            break
    return tuple(out)


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
        # Same surface can appear on multiple vocab rows (e.g. different POS passes);
        # keep one row per clue surface — highest Zipf wins (commonest reading).
        best: dict[str, tuple[str, float, int]] = {}
        for surface, lemma, zipf in zip(
            df["surface"].tolist(),
            df["lemma"].tolist(),
            df["zipf"].tolist(),
            strict=True,
        ):
            idx = matrix.index_of(surface)
            if idx is None:
                continue
            z = float(zipf)
            prev = best.get(surface)
            if prev is None or z > prev[1]:
                best[surface] = (lemma, z, idx)
        ordered = sorted(best.items(), key=lambda kv: kv[0])
        keep_surfaces = [s for s, _ in ordered]
        keep_lemmas = [t[0] for _, t in ordered]
        keep_zipfs = [t[1] for _, t in ordered]
        keep_idx = [t[2] for _, t in ordered]
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
         rule. Rank survivors by Monte Carlo expected reward (EV).
      4. Pick the highest-scoring candidate; keep the top `top_k` distinct clue
         surfaces for the trace (best-scoring N per clue).
    """

    def __init__(
        self,
        matrix: EmbeddingMatrix,
        clue_vocabulary: Vocabulary,
        *,
        risk: float = 0.5,
        weights: ScoringWeights | None = None,
        rule_strictness: RuleStrictness = "lemma_substring",
        top_k: int = 200,
        reranker: SpymasterReranker | None = None,
        clue_surface_exclusions: frozenset[str] = frozenset(),
    ) -> None:
        self.matrix = matrix
        self.clue_vocabulary = clue_vocabulary
        self.weights = weights or ScoringWeights.from_risk(risk)
        self.rule_strictness = rule_strictness
        self.top_k = top_k
        self.reranker = reranker
        self._clue_surface_exclusions = clue_surface_exclusions
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
        sim_all = clue_vecs @ self._stack(active).T  # (M, A)

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

        forbidden_surfaces = self._clue_surface_exclusions | view.prior_clue_words

        candidates, veto_count, illegal_count = self._score_all(
            clue_idx=clue_idx,
            team=team,
            friendly=friendly,
            sort_idx=sort_idx,
            sorted_friendly=sorted_friendly,
            best_non_friendly=best_non_friendly,
            sim_assassin=sim_assassin,
            sim_all=sim_all,
            active_cards=active,
            forbidden_surfaces=forbidden_surfaces,
        )

        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            logger.debug(
                "spymaster embedding ranking: 0 survivors "
                "(vetoes=%d, illegal_clue_slots=%d)",
                veto_count,
                illegal_count,
            )
        else:
            n_show = min(
                len(candidates),
                (
                    self.reranker.top_k
                    if self.reranker is not None
                    else min(self.top_k, 40)
                ),
            )
            logger.debug(
                "spymaster embedding ranking: %d survivors (vetoes=%d, "
                "illegal_clue_slots=%d); top %d by embedding score:",
                len(candidates),
                veto_count,
                illegal_count,
                n_show,
            )
            for rank, c in enumerate(candidates[:n_show], start=1):
                comp = c.components
                logger.debug(
                    "  emb #%d ev=%.4f margin=%.4f clue=%r n=%d targets=%s | "
                    "friendly_min_sim=%.4f zipf=%.2f",
                    rank,
                    c.embedding_score,
                    c.margin,
                    c.clue,
                    c.n,
                    list(c.targets),
                    comp.friendly_min_sim,
                    c.zipf,
                )

        if self.reranker is not None and candidates:
            # Only the top-K embedding shortlist is reranked. The tail is
            # discarded — final blended scores mix EV with ``LLM * N_eff`` and
            # are not comparable to unrouted candidates that only have EV.
            shortlist = self._build_rerank_shortlist(
                candidates, top_k=self.reranker.top_k
            )
            candidates = list(self.reranker.rerank(shortlist, view))
            candidates.sort(key=lambda c: c.score, reverse=True)

            n_final = min(len(candidates), self.reranker.top_k)
            logger.info(
                "spymaster ranking after rerank: top %d by blended score:",
                n_final,
            )
            for rank, c in enumerate(candidates[:n_final], start=1):
                llm = f"{c.llm_score:.4f}" if c.llm_score is not None else "—"
                reason = (c.llm_reason or "").strip()
                tail = f" | {reason}" if reason else ""
                logger.info(
                    "  final #%d blend=%.4f emb=%.4f llm=%s clue=%r n=%d targets=%s%s",
                    rank,
                    c.score,
                    c.embedding_score,
                    llm,
                    c.clue,
                    c.n,
                    list(c.targets),
                    tail,
                )

        if not candidates:
            raise NoLegalClueError(
                f"no candidates passed vetoes "
                f"(rejected: {veto_count} by margin/assassin, {illegal_count} by legality)"
            )

        chosen = candidates[0]
        top = _dedupe_candidates_by_clue(candidates, limit=self.top_k)

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

    def _build_rerank_shortlist(
        self, candidates: list[Candidate], *, top_k: int
    ) -> list[Candidate]:
        """Top ``top_k`` candidates by expected reward (one row per ``(clue, N)``)."""
        if not candidates or top_k <= 0:
            return []

        ordered = sorted(
            candidates,
            key=lambda c: (
                c.components.expected_reward_raw,
                c.embedding_score,
                c.margin,
            ),
            reverse=True,
        )
        chosen: list[Candidate] = []
        chosen_keys: set[tuple[str, int]] = set()
        for cand in ordered:
            if len(chosen) >= top_k:
                break
            key = (cand.clue, cand.n)
            if key in chosen_keys:
                continue
            chosen.append(cand)
            chosen_keys.add(key)
        return chosen

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
        team: Color,
        friendly: list[Card],
        sort_idx: np.ndarray,
        sorted_friendly: np.ndarray,
        best_non_friendly: np.ndarray,
        sim_assassin: np.ndarray,
        sim_all: np.ndarray,
        active_cards: list[Card],
        forbidden_surfaces: frozenset[str],
    ) -> tuple[list[Candidate], int, int]:
        weights = self.weights
        F = sorted_friendly.shape[1]
        max_n = min(F, max(1, int(weights.lane_max_n)))
        candidates: list[Candidate] = []
        veto_count = 0
        illegal_count = 0

        seeds: dict[tuple[str, int], int] = {}
        for i in range(len(clue_idx.surfaces)):
            surface = clue_idx.surfaces[i]
            lemma = clue_idx.lemmas[i]
            zipf = float(clue_idx.zipfs[i])

            assassin_sim = float(sim_assassin[i])
            row_sorted = sorted_friendly[i]
            row_all = sim_all[i]

            # Legality (filters surface and lemma vs all active cards).
            if not is_legal_clue(
                clue_surface=surface,
                clue_lemma=lemma,
                active_cards=active_cards,
                strictness=self.rule_strictness,
                forbidden_surfaces=forbidden_surfaces,
            ):
                illegal_count += max_n  # one illegal clue → rejected for each ``N``
                continue

            for n in range(1, max_n + 1):
                friendly_min_sim = float(row_sorted[n - 1])
                margin = friendly_min_sim - float(best_non_friendly[i])

                if margin < weights.margin_floor:
                    veto_count += 1
                    continue
                if assassin_sim > weights.assassin_ceiling:
                    veto_count += 1
                    continue

                seed = (i + 1) * 1009 + n * 9173
                exp_reward_raw = self._estimate_expected_reward(
                    similarities=row_all,
                    active_cards=active_cards,
                    team=team,
                    n=n,
                    seed=seed,
                    trials=weights.adaptive_mc_base_trials,
                )
                components = ScoreComponents(
                    expected_reward_raw=exp_reward_raw,
                    friendly_min_sim=friendly_min_sim,
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
                seeds[(surface, n)] = seed
        if candidates and self.weights.adaptive_mc_extra_trials > 0:
            candidates = self._refine_mc_estimates(
                candidates=candidates,
                seeds=seeds,
                similarities_by_clue={
                    clue_idx.surfaces[i]: sim_all[i]
                    for i in range(len(clue_idx.surfaces))
                },
                active_cards=active_cards,
                team=team,
            )
        return candidates, veto_count, illegal_count

    def _refine_mc_estimates(
        self,
        *,
        candidates: list[Candidate],
        seeds: dict[tuple[str, int], int],
        similarities_by_clue: dict[str, np.ndarray],
        active_cards: list[Card],
        team: Color,
    ) -> list[Candidate]:
        """Run extra MC trials for candidates close to the global EV leader."""
        w = self.weights
        out = list(candidates)
        if not out:
            return out
        best_ev = max(c.components.expected_reward_raw for c in out)
        for idx, cand in enumerate(out):
            if best_ev - cand.components.expected_reward_raw > w.adaptive_mc_ev_band:
                continue
            sims = similarities_by_clue.get(cand.clue)
            seed = seeds.get((cand.clue, cand.n))
            if sims is None or seed is None:
                continue
            extra_ev = self._estimate_expected_reward(
                similarities=sims,
                active_cards=active_cards,
                team=team,
                n=cand.n,
                seed=seed + 37,
                trials=w.adaptive_mc_extra_trials,
            )
            base_trials = max(1, int(w.adaptive_mc_base_trials))
            extra_trials = max(1, int(w.adaptive_mc_extra_trials))
            blended_ev = (
                cand.components.expected_reward_raw * base_trials
                + extra_ev * extra_trials
            ) / float(base_trials + extra_trials)
            new_components = replace(
                cand.components,
                expected_reward_raw=blended_ev,
            )
            out[idx] = replace(
                cand,
                components=new_components,
                embedding_score=new_components.total,
                score=new_components.total,
            )
        return out

    def _estimate_expected_reward(
        self,
        *,
        similarities: np.ndarray,
        active_cards: list[Card],
        team: Color,
        n: int,
        seed: int,
        trials: int | None = None,
    ) -> float:
        """Estimate expected clue value via Monte Carlo over softmax sampling.

        Pick distribution is softmax over cosine similarities scaled by ``mc_temperature``.
        Optionally ``mc_rank_bias`` lowers probability for worse-by-sim rank (even when gaps
        are tiny), yielding **stochastic rollouts biased toward greedy order** —
        sharper than similarity-only softmax alone.
        """
        w = self.weights
        if not active_cards:
            return 0.0
        temp = max(1e-3, float(w.mc_temperature))
        rank_bias = max(0.0, float(w.mc_rank_bias))
        sims = similarities.astype(np.float64)
        trials = max(1, int(trials if trials is not None else w.mc_trials))
        rng = np.random.default_rng(seed)
        opp = team.opponent()
        rewards = {
            team: float(w.reward_friendly),
            opp: float(w.reward_opponent),
            Color.NEUTRAL: float(w.reward_neutral),
            Color.ASSASSIN: float(w.reward_assassin),
        }

        total = 0.0
        for _ in range(trials):
            avail = np.ones(len(active_cards), dtype=bool)
            trial_score = 0.0
            picks = 0
            while picks < n and np.any(avail):
                avail_idx = np.flatnonzero(avail)
                logits = sims[avail_idx] / temp
                if rank_bias > 0.0 and len(avail_idx) > 1:
                    order = np.argsort(-sims[avail_idx])
                    rk = np.empty(len(avail_idx), dtype=np.float64)
                    rk[order] = np.arange(len(avail_idx), dtype=np.float64)
                    logits -= rank_bias * np.log1p(rk)
                logits -= logits.max()
                probs = np.exp(logits)
                probs_sum = probs.sum()
                if probs_sum <= 0.0:
                    probs = np.full_like(probs, 1.0 / len(probs))
                else:
                    probs /= probs_sum
                chosen_local = int(rng.choice(len(avail_idx), p=probs))
                idx = int(avail_idx[chosen_local])
                avail[idx] = False
                card = active_cards[idx]
                if card.color == team:
                    trial_score += rewards[team]
                    picks += 1
                    continue
                if card.color == opp:
                    trial_score += rewards[opp]
                    break
                if card.color == Color.NEUTRAL:
                    trial_score += rewards[Color.NEUTRAL]
                    break
                trial_score += rewards[Color.ASSASSIN]
                break
            total += trial_score
        return total / float(trials)
