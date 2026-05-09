"""Game-aware effective risk from board-objective counts (deterministic pure functions).

Δ = opponent unrevealed team cards − our unrevealed team cards.

effective_risk = clamp(base_risk · exp(−s·Δ), r_min, r_max)

Per-parameter modulation (baseline knobs from YAML / ``from_risk``):

``value ← clamp(baseline · exp(direction_k · β_k · (effective_risk − base_risk)), ...)``

Pivoting on ``base_risk`` (not 0.5) ensures Δ=0 ⇒ effective_risk=base_risk ⇒ multipliers are 1 and
behaviour matches static ``base_risk`` baselines (issue #1 user story 14).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, replace

from codenames_ai.agent.interfaces import Guesser, Spymaster
from codenames_ai.agent.scoring import ScoringWeights, StopPolicy
from codenames_ai.agent.trace import GuesserTrace, RiskSnapshot, SpymasterTrace
from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.game.models import Board, Color, Clue, GuesserView, SpymasterView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DynamicRiskPolicy:
    """Runtime policy copied from eval YAML ``dynamic_risk`` (strict keys validated there)."""

    enabled: bool = False
    s: float = 0.12
    min_risk: float = 0.0
    max_risk: float = 1.0
    beta_margin_floor: float = 0.25
    beta_assassin_ceiling: float = 0.25
    beta_confidence_floor: float = 0.25
    beta_bonus_gap: float = 0.25


def unrevealed_objective_count(board: Board, team: Color) -> int:
    return sum(1 for c in board.cards if c.color == team and not c.revealed)


def objectives_delta(board: Board, team: Color) -> tuple[int, int, float]:
    opp = team.opponent()
    ours = unrevealed_objective_count(board, team)
    theirs = unrevealed_objective_count(board, opp)
    return ours, theirs, float(theirs - ours)


def compute_effective_risk(base_risk: float, delta: float, policy: DynamicRiskPolicy) -> float:
    if not policy.enabled:
        return float(base_risk)
    r0 = max(0.0, min(1.0, float(base_risk)))
    eff = r0 * math.exp(-float(policy.s) * float(delta))
    return max(float(policy.min_risk), min(float(policy.max_risk), eff))


def risk_snapshot_for_board(
    board: Board,
    team: Color,
    *,
    base_risk: float,
    policy: DynamicRiskPolicy,
) -> RiskSnapshot:
    ours, theirs, delta = objectives_delta(board, team)
    eff = compute_effective_risk(base_risk, delta, policy)
    return RiskSnapshot(
        base_risk=float(base_risk),
        effective_risk=float(eff),
        delta_objectives=float(delta),
        ours_unrevealed=int(ours),
        theirs_unrevealed=int(theirs),
        dynamic_enabled=bool(policy.enabled),
    )


# Hardcoded directions: +1 = higher effective_risk increases scalar; −1 = decreases.
_DIR_MARGIN_FLOOR = -1.0
_DIR_ASSASSIN_CEILING = 1.0
_DIR_CONFIDENCE_FLOOR = -1.0
_DIR_BONUS_GAP = 1.0

# Conservative clamps so modulation cannot move values into absurd ranges.
_CLAMP_MARGIN = (0.0, 0.22)
_CLAMP_ASSASSIN = (0.12, 0.55)
_CLAMP_CONF_FLOOR = (-1.1, 0.40)
_CLAMP_BONUS_GAP = (-1.1, 0.35)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _nudge(
    baseline: float,
    *,
    direction: float,
    beta: float,
    base_risk: float,
    effective_risk: float,
    lo: float,
    hi: float,
) -> float:
    drift = float(effective_risk) - float(base_risk)
    mult = math.exp(direction * float(beta) * drift)
    return _clamp(float(baseline) * mult, lo, hi)


def modulate_scoring_weights(
    weights: ScoringWeights,
    *,
    base_risk: float,
    effective_risk: float,
    policy: DynamicRiskPolicy,
) -> ScoringWeights:
    if not policy.enabled:
        return weights
    br, er = float(base_risk), float(effective_risk)
    return replace(
        weights,
        margin_floor=_nudge(
            weights.margin_floor,
            direction=_DIR_MARGIN_FLOOR,
            beta=policy.beta_margin_floor,
            base_risk=br,
            effective_risk=er,
            lo=_CLAMP_MARGIN[0],
            hi=_CLAMP_MARGIN[1],
        ),
        assassin_ceiling=_nudge(
            weights.assassin_ceiling,
            direction=_DIR_ASSASSIN_CEILING,
            beta=policy.beta_assassin_ceiling,
            base_risk=br,
            effective_risk=er,
            lo=_CLAMP_ASSASSIN[0],
            hi=_CLAMP_ASSASSIN[1],
        ),
    )


def modulate_stop_policy(
    stop_policy: StopPolicy,
    *,
    base_risk: float,
    effective_risk: float,
    policy: DynamicRiskPolicy,
) -> StopPolicy:
    if not policy.enabled:
        return stop_policy
    br, er = float(base_risk), float(effective_risk)
    return replace(
        stop_policy,
        confidence_floor=_nudge(
            stop_policy.confidence_floor,
            direction=_DIR_CONFIDENCE_FLOOR,
            beta=policy.beta_confidence_floor,
            base_risk=br,
            effective_risk=er,
            lo=_CLAMP_CONF_FLOOR[0],
            hi=_CLAMP_CONF_FLOOR[1],
        ),
        bonus_gap_threshold=_nudge(
            stop_policy.bonus_gap_threshold,
            direction=_DIR_BONUS_GAP,
            beta=policy.beta_bonus_gap,
            base_risk=br,
            effective_risk=er,
            lo=_CLAMP_BONUS_GAP[0],
            hi=_CLAMP_BONUS_GAP[1],
        ),
        risk=_clamp(er, 0.0, 1.0),
    )


def apply_effective_risk(
    weights: ScoringWeights,
    stop_policy: StopPolicy,
    *,
    base_risk: float,
    effective_risk: float,
    policy: DynamicRiskPolicy,
) -> tuple[ScoringWeights, StopPolicy]:
    """Return modulated copies; when ``policy.enabled`` is false, inputs are returned unchanged."""
    return (
        modulate_scoring_weights(
            weights,
            base_risk=base_risk,
            effective_risk=effective_risk,
            policy=policy,
        ),
        modulate_stop_policy(
            stop_policy,
            base_risk=base_risk,
            effective_risk=effective_risk,
            policy=policy,
        ),
    )


class DynamicRiskAISpymaster(Spymaster):
    """Wraps ``AISpymaster``; restores baseline weights after each ``give_clue``."""

    __slots__ = ("_inner", "_base_risk", "_policy", "_lock")

    def __init__(
        self,
        inner: AISpymaster,
        *,
        base_risk: float,
        policy: DynamicRiskPolicy,
    ) -> None:
        self._inner = inner
        self._base_risk = float(base_risk)
        self._policy = policy
        self._lock = threading.Lock()

    @property
    def baseline_weights(self) -> ScoringWeights:
        return self._inner.weights

    def give_clue(self, view: SpymasterView) -> SpymasterTrace:
        with self._lock:
            snap = risk_snapshot_for_board(
                view.board, view.team, base_risk=self._base_risk, policy=self._policy
            )
            logger.info(
                "effective risk before spymaster (team=%s): base=%.4f effective=%.4f "
                "Δ_objectives=%+.0f unrevealed_ours/theirs=%d/%d",
                view.team.value,
                snap.base_risk,
                snap.effective_risk,
                snap.delta_objectives,
                snap.ours_unrevealed,
                snap.theirs_unrevealed,
            )
            w0 = self._inner.weights
            w_eff = modulate_scoring_weights(
                w0,
                base_risk=self._base_risk,
                effective_risk=snap.effective_risk,
                policy=self._policy,
            )
            self._inner.weights = w_eff
            try:
                trace = self._inner.give_clue(view)
            finally:
                self._inner.weights = w0
            return replace(trace, risk_snapshot=snap, weights=w_eff)


class DynamicRiskAIGuesser(Guesser):
    """Wraps ``AIGuesser``; restores baseline ``stop_policy`` after each ``guess``."""

    __slots__ = ("_inner", "_base_risk", "_policy", "_lock")

    def __init__(
        self,
        inner: AIGuesser,
        *,
        base_risk: float,
        policy: DynamicRiskPolicy,
    ) -> None:
        self._inner = inner
        self._base_risk = float(base_risk)
        self._policy = policy
        self._lock = threading.Lock()

    @property
    def baseline_stop_policy(self) -> StopPolicy:
        return self._inner.stop_policy

    def guess(self, view: GuesserView, clue: Clue) -> GuesserTrace:
        with self._lock:
            snap = risk_snapshot_for_board(
                view.board, view.team, base_risk=self._base_risk, policy=self._policy
            )
            clue_tag = "(pass)" if clue.is_pass() else repr(clue.word)
            logger.info(
                "effective risk before guesser (team=%s clue=%s): base=%.4f effective=%.4f "
                "Δ_objectives=%+.0f unrevealed_ours/theirs=%d/%d",
                view.team.value,
                clue_tag,
                snap.base_risk,
                snap.effective_risk,
                snap.delta_objectives,
                snap.ours_unrevealed,
                snap.theirs_unrevealed,
            )
            s0 = self._inner.stop_policy
            s_eff = modulate_stop_policy(
                s0,
                base_risk=self._base_risk,
                effective_risk=snap.effective_risk,
                policy=self._policy,
            )
            self._inner.stop_policy = s_eff
            try:
                trace = self._inner.guess(view, clue)
            finally:
                self._inner.stop_policy = s0
            return replace(trace, risk_snapshot=snap, stop_policy=s_eff)
