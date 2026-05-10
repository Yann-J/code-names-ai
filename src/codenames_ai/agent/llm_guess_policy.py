"""Pure policy functions for the LLM-primary guesser.

Inputs are normalised score maps (``fit``, ``danger`` keyed by every
unrevealed board word) plus a small ``ContinueGate`` config. Outputs are
deterministic: a single committed word per turn step plus a continue/stop
decision the orchestrator uses to schedule the next physical guess.

No LLM calls or game-state mutation happen here — keeps the engine trivial
to unit-test against golden score fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinueGate:
    """Numeric guard rails layered on top of the LLM's continue/stop flag.

    The orchestrator must already verify game-rule legality (at least one
    successful guess this clue, attempts remaining, game still active) before
    consulting the gate. The gate then enforces minimum confidence so a
    ``continue=true`` from the model cannot override collapsed scores.
    """

    min_combined: float = 0.0
    """Combined score (``fit − λ·danger``) of the next pick must be ≥ this."""

    min_margin_to_second: float = 0.0
    """Gap between top combined score and second place must be ≥ this."""

    min_fit: float = 0.0
    """Raw fit of the next pick must be ≥ this (assassin-style sanity floor)."""


def combined_scores(
    fit: dict[str, float],
    danger: dict[str, float],
    *,
    lambda_danger: float,
) -> dict[str, float]:
    """``combined[w] = fit[w] − λ·danger[w]`` for every shared word.

    Words missing from either map are dropped from the output (parser already
    enforces matching key sets in the LLM-primary path; this guard keeps the
    function safe for fallback paths).
    """
    out: dict[str, float] = {}
    for word, f in fit.items():
        d = danger.get(word)
        if d is None:
            continue
        out[word] = float(f) - float(lambda_danger) * float(d)
    return out


def argmax_combined(
    combined: dict[str, float],
    *,
    candidates: tuple[str, ...] | None = None,
) -> str:
    """Pick the highest-scoring word; ties broken by the candidate ordering.

    ``candidates`` (when provided) restricts the search and supplies a stable
    iteration order for reproducible tie-breaks. Without it, the function falls
    back to lexicographic order.
    """
    if not combined:
        raise ValueError("argmax_combined requires at least one scored word")
    if candidates:
        ordered = [w for w in candidates if w in combined]
        if not ordered:
            raise ValueError("none of the supplied candidates carry a score")
    else:
        ordered = sorted(combined.keys())

    best = ordered[0]
    best_score = combined[best]
    for w in ordered[1:]:
        s = combined[w]
        if s > best_score:
            best = w
            best_score = s
    return best


def margin_to_second(combined: dict[str, float], chosen: str) -> float:
    """Best minus second-best combined score (0 when only one card remains)."""
    if chosen not in combined or len(combined) <= 1:
        return 0.0
    top = combined[chosen]
    second = max(s for w, s in combined.items() if w != chosen)
    return float(top - second)


@dataclass(frozen=True)
class ContinueDecision:
    """Outcome of the two-layer continue/stop gate."""

    proceed: bool
    reason: str  # one of: 'llm_stop' | 'min_combined' | 'min_margin' | 'min_fit' | 'no_more_unrevealed' | 'gate_passed' | 'attempts_exhausted'


def evaluate_continue_gate(
    *,
    llm_continue: bool,
    chosen: str,
    next_combined: dict[str, float],
    next_fit: dict[str, float],
    gate: ContinueGate,
    attempts_remaining_after: int | None,
) -> ContinueDecision:
    """Apply the two-layer rule: honour ``continue`` only when guards agree.

    Caller passes the ``next_*`` score maps (i.e. the maps that *would* drive
    the following physical guess); for the immediate post-guess decision these
    typically come from the same LLM call (after dropping the just-chosen
    word). ``chosen`` is the word that was just committed and is provided
    purely for traceability — the gate inspects the *next* candidate (top of
    ``next_combined``). ``attempts_remaining_after`` is the count after the
    just-committed pick: ``None`` means uncapped, ``<= 0`` ends the turn.
    """
    del chosen  # reserved for future logging hooks; gate uses next_* only
    if not llm_continue:
        return ContinueDecision(proceed=False, reason="llm_stop")
    if attempts_remaining_after is not None and attempts_remaining_after <= 0:
        return ContinueDecision(proceed=False, reason="attempts_exhausted")
    if not next_combined:
        return ContinueDecision(proceed=False, reason="no_more_unrevealed")
    next_top = max(next_combined, key=lambda w: next_combined[w])
    if next_combined[next_top] < gate.min_combined:
        return ContinueDecision(proceed=False, reason="min_combined")
    if next_fit:
        top_fit = next_fit.get(next_top, max(next_fit.values()))
        if top_fit < gate.min_fit:
            return ContinueDecision(proceed=False, reason="min_fit")
    if len(next_combined) >= 2:
        margin = margin_to_second(next_combined, next_top)
        if margin < gate.min_margin_to_second:
            return ContinueDecision(proceed=False, reason="min_margin")
    return ContinueDecision(proceed=True, reason="gate_passed")
