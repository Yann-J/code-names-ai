"""LLM-primary guesser: one LLM call per physical guess + deterministic policy.

Implements the Code Names guesser as a per-step loop driven by the
orchestrator's ``play_turn`` callback:

  1. Build a fresh prompt from the current ``GuesserView`` and compressed
     dual-team history.
  2. Issue one LLM call (schema-first; prompt-only JSON fallback) at
     temperature 0 with one automatic retry.
  3. Compute ``combined = fit − λ·danger`` over every unrevealed word and
     commit ``argmax(combined)``.
  4. Reveal via the orchestrator callback. If the engine signals the turn
     ended (wrong colour, attempts exhausted, game over), stop.
  5. Otherwise apply the two-layer continue gate (model's ``continue`` flag +
     numeric guards) and either loop or stop.

On parse failure (after the configured retry count), the adapter falls back
to deterministic embedding cosine argmax when an ``EmbeddingMatrix`` covers
the clue and every unrevealed word; otherwise it picks uniformly at random.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from codenames_ai.agent.interfaces import Guesser, RevealOutcome
from codenames_ai.agent.llm_guess_policy import (
    ContinueGate,
    argmax_combined,
    combined_scores,
    evaluate_continue_gate,
    margin_to_second,
)
from codenames_ai.agent.llm_guess_scorer import (
    CompressedTurn,
    LLMGuessScorer,
    LLMScores,
    StepEnvelope,
    build_compressed_history,
)
from codenames_ai.agent.scoring import StopPolicy
from codenames_ai.agent.trace import CandidateGuess, GuesserTrace, LLMGuessStep
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Clue, GuesserView
from codenames_ai.game.state import TurnEvent

logger = logging.getLogger(__name__)


class LLMGuesser(Guesser):
    """Iterative, LLM-primary guesser implementing the issue #3 PRD.

    The orchestrator drives the turn through ``play_turn``, supplying a fresh
    ``GuesserView`` after each reveal. Outside the orchestrator, the legacy
    ``guess()`` API runs the same loop without revealing — useful for offline
    inspection: it commits picks until the gate stops it or the trace would
    contradict the LLM's stop signal, then returns the accumulated trace
    without mutating any board.
    """

    def __init__(
        self,
        scorer: LLMGuessScorer,
        *,
        lambda_danger: float = 0.5,
        gate: ContinueGate | None = None,
        embedding_matrix: EmbeddingMatrix | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if lambda_danger < 0.0:
            raise ValueError(f"lambda_danger must be >= 0, got {lambda_danger}")
        self.scorer = scorer
        self.lambda_danger = float(lambda_danger)
        self.gate = gate or ContinueGate()
        self.embedding_matrix = embedding_matrix
        self.rng = rng or np.random.default_rng()

    def guess(self, view: GuesserView, clue: Clue) -> GuesserTrace:
        """Single-call inspection mode (no real reveals).

        Useful for tests / analysis: produces a trace as if every guess had
        been the (LLM-suggested) committed pick. Picks are removed from the
        scored set between iterations to simulate "next-best argmax" without
        re-prompting; in real game play the orchestrator's ``play_turn``
        loop is the authoritative path.
        """
        return self._run(view, clue, history=(), reveal=None)

    def play_turn(
        self,
        view: GuesserView,
        clue: Clue,
        history: tuple[TurnEvent, ...],
        *,
        reveal: Callable[[str], RevealOutcome],
    ) -> GuesserTrace:
        compressed = build_compressed_history(history, omit_current_clue=True)
        return self._run(view, clue, history=compressed, reveal=reveal)

    def _run(
        self,
        view: GuesserView,
        clue: Clue,
        *,
        history: tuple[CompressedTurn, ...],
        reveal: Callable[[str], RevealOutcome] | None,
    ) -> GuesserTrace:
        if clue.is_pass():
            return GuesserTrace(
                candidates=(),
                guesses=(),
                stop_policy=self._diagnostic_stop_policy(),
                bonus_attempted=False,
                stop_reason="pass_clue",
            )

        committed_words: list[str] = []
        steps: list[LLMGuessStep] = []
        last_step_candidates: tuple[CandidateGuess, ...] = ()
        stop_reason = "no_more_unrevealed"
        bonus_attempted = False

        # Inspection mode (no orchestrator) needs to track picks locally so
        # subsequent iterations don't re-pick the same word.
        inspection_excluded: set[str] = set()
        current_view = view

        while True:
            unrevealed = tuple(c.word for c in current_view.board.active())
            unrevealed_now = tuple(w for w in unrevealed if w not in inspection_excluded)
            if not unrevealed_now:
                stop_reason = "no_more_unrevealed"
                break

            scores, envelope = self.scorer.score(
                view=current_view, clue=clue, history=history
            )
            fit_map, danger_map, llm_continue, fallback_path = self._resolve_scores(
                scores=scores,
                envelope=envelope,
                view=current_view,
                clue=clue,
                excluded=inspection_excluded,
            )
            if not fit_map:
                stop_reason = fallback_path or "no_more_unrevealed"
                break

            combined = combined_scores(
                fit_map, danger_map, lambda_danger=self.lambda_danger
            )
            chosen = argmax_combined(combined, candidates=unrevealed_now)
            margin = margin_to_second(combined, chosen)

            # Build a candidate snapshot of this step (descending by combined).
            last_step_candidates = self._candidates_from_scores(
                fit_map=fit_map,
                combined=combined,
                committed_word=chosen,
                clue_count=clue.count,
                step_index=len(steps),
            )

            # Evaluate gate using the post-commit score map (drop chosen word).
            next_combined = {w: s for w, s in combined.items() if w != chosen}
            next_fit = {w: s for w, s in fit_map.items() if w != chosen}
            attempts_after: int | None = None  # orchestrator handles attempts
            gate_result = evaluate_continue_gate(
                llm_continue=llm_continue,
                chosen=chosen,
                next_combined=next_combined,
                next_fit=next_fit,
                gate=self.gate,
                attempts_remaining_after=attempts_after,
            )

            steps.append(
                LLMGuessStep(
                    guess=chosen,
                    fit=dict(fit_map),
                    danger=dict(danger_map),
                    combined=dict(combined),
                    lambda_danger=self.lambda_danger,
                    continue_flag=llm_continue,
                    continue_gate_passed=gate_result.proceed,
                    gate_reason=gate_result.reason,
                    fallback_path=fallback_path,
                    model_id=envelope.model_id,
                    schema_used=envelope.schema_used,
                    raw_response_hash=envelope.raw_response_hash,
                    margin_to_second=margin,
                )
            )
            committed_words.append(chosen)

            if reveal is None:
                # Inspection mode: simulate commit without orchestrator reveal.
                inspection_excluded.add(chosen)
                if len(committed_words) > clue.count:
                    bonus_attempted = True
                if not gate_result.proceed:
                    stop_reason = self._stop_reason_for(gate_result.reason)
                    break
                if not next_combined:
                    stop_reason = "no_more_unrevealed"
                    break
                continue

            outcome = reveal(chosen)
            if outcome.game_over:
                stop_reason = "game_over"
                break
            if outcome.turn_ended:
                stop_reason = "turn_ended_by_engine"
                break
            if not gate_result.proceed:
                stop_reason = self._stop_reason_for(gate_result.reason)
                break

            if len(committed_words) >= clue.count + 1:
                bonus_attempted = True

            current_view = outcome.view
            if current_view is None:
                stop_reason = "turn_ended_by_engine"
                break

        return GuesserTrace(
            candidates=last_step_candidates,
            guesses=tuple(committed_words),
            stop_policy=self._diagnostic_stop_policy(),
            bonus_attempted=bonus_attempted,
            stop_reason=stop_reason,
            llm_steps=tuple(steps),
        )

    def _resolve_scores(
        self,
        *,
        scores: LLMScores | None,
        envelope: StepEnvelope,
        view: GuesserView,
        clue: Clue,
        excluded: set[str],
    ) -> tuple[dict[str, float], dict[str, float], bool, str]:
        """Return (fit, danger, continue, fallback_path) for one physical step."""
        if scores is not None:
            fit = {w: v for w, v in scores.fit.items() if w not in excluded}
            danger = {w: v for w, v in scores.danger.items() if w not in excluded}
            return fit, danger, scores.continue_flag, envelope.fallback_path

        # LLM parse failed — try embedding fallback when coverage is complete.
        unrevealed = tuple(
            c.word for c in view.board.active() if c.word not in excluded
        )
        emb = self._embedding_fallback(clue=clue, unrevealed=unrevealed)
        if emb is not None:
            fit_e, danger_e = emb
            return fit_e, danger_e, False, "embedding_fallback"

        if not unrevealed:
            return {}, {}, False, "no_unrevealed"

        # Uniform dead-end fallback: random pick, danger flat at 0.5.
        idx = int(self.rng.integers(0, len(unrevealed)))
        chosen = unrevealed[idx]
        fit_u = {w: (1.0 if w == chosen else 0.0) for w in unrevealed}
        danger_u = {w: 0.5 for w in unrevealed}
        logger.warning(
            "LLM-primary guesser: uniform dead-end fallback fired for clue=%r (chose %r)",
            clue.word,
            chosen,
        )
        return fit_u, danger_u, False, "uniform_dead_end"

    def _embedding_fallback(
        self,
        *,
        clue: Clue,
        unrevealed: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Deterministic cosine-argmax fit over unrevealed words; danger left flat.

        Returns ``None`` when the embedding matrix is missing or fails to
        cover the clue word + every unrevealed word (so the caller can route
        to the uniform dead-end path).
        """
        matrix = self.embedding_matrix
        if matrix is None:
            return None
        if not unrevealed:
            return None
        if clue.word not in matrix:
            return None
        if any(w not in matrix for w in unrevealed):
            return None
        clue_vec = matrix[clue.word]
        idx = np.array([matrix.surface_to_index[w] for w in unrevealed])
        sims = (matrix.vectors[idx] @ clue_vec).astype(np.float32)
        # Map raw cosine [-1, 1] into [0, 1] for the fit channel.
        fit_vals = ((sims + 1.0) * 0.5).clip(0.0, 1.0)
        fit = {w: float(v) for w, v in zip(unrevealed, fit_vals)}
        danger = {w: 0.0 for w in unrevealed}
        return fit, danger

    def _candidates_from_scores(
        self,
        *,
        fit_map: dict[str, float],
        combined: dict[str, float],
        committed_word: str,
        clue_count: int,
        step_index: int,
    ) -> tuple[CandidateGuess, ...]:
        """Build a ``CandidateGuess`` snapshot, sorted by combined score desc."""
        order = sorted(combined.keys(), key=lambda w: combined[w], reverse=True)
        out: list[CandidateGuess] = []
        for rank, word in enumerate(order):
            out.append(
                CandidateGuess(
                    word=word,
                    similarity=float(fit_map.get(word, 0.0)),
                    score=float(combined[word]),
                    rank=rank,
                    committed=(word == committed_word),
                    is_bonus=(word == committed_word and step_index >= clue_count),
                    llm_score=float(fit_map.get(word, 0.0)),
                    llm_reason=None,
                )
            )
        return tuple(out)

    @staticmethod
    def _stop_reason_for(gate_reason: str) -> str:
        # Map policy gate reasons into the broader trace ``stop_reason`` namespace.
        return f"llm_gate:{gate_reason}"

    def _diagnostic_stop_policy(self) -> StopPolicy:
        """Synthesise a ``StopPolicy`` view of the gate for trace consumers.

        ``StopPolicy`` is the established structure other code (analysis UI,
        risk wrappers) reads. We expose ``min_combined`` as a "confidence
        floor" surrogate and disable the bonus channel: the LLM-primary policy
        does not have a separate N+1 mechanism (the gate decides each step).
        """
        return StopPolicy(
            confidence_floor=float(self.gate.min_combined),
            bonus_gap_threshold=-1.0,
            risk=0.5,
        )
