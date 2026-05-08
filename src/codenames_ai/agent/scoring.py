from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Tunable weights for the spymaster scoring formula.

    Most users should construct this via `from_risk()`. Fields are all directly
    overridable for experimentation.
    """

    margin_floor: float
    """Hard veto: reject candidates whose margin is below this."""

    assassin_ceiling: float
    """Hard veto: reject if assassin similarity exceeds this."""

    ambition_weight: float
    """Bonus per additional target word in the subset (multiplied by N-1)."""

    margin_weight: float
    """Multiplier on margin for the soft scoring component."""

    freq_weight: float
    """Scale of the saturating frequency bonus."""

    assassin_weight: float
    """Penalty scale on assassin similarity (beyond the hard veto)."""

    opponent_weight: float
    """Penalty multiplier on the worst (max) opponent-card similarity."""

    prefer_min_targets: int
    """Soft floor on friendly count each clue should cover (if that many exist)."""

    undercluster_penalty_weight: float
    """Penalty per target below ``min(prefer_min_targets, F)``."""

    expected_reward_weight: float
    """Multiplier on Monte Carlo expected reward (friendly gain vs risk)."""

    mc_trials: int
    """Number of Monte Carlo rollouts used per `(clue, N)` candidate."""

    mc_temperature: float
    """Softmax temperature used to turn similarities into pick likelihoods."""

    reward_friendly: float
    reward_neutral: float
    reward_opponent: float
    reward_assassin: float
    """Per-pick rewards used by Monte Carlo simulation."""

    @classmethod
    def from_risk(cls, risk: float) -> "ScoringWeights":
        """Map a single `risk` scalar in [0, 1] to a weight set.

        risk = 0 → cautious: high margins, low ambition, strong assassin penalties.
        risk = 1 → aggressive: thin margins, ambition rewarded.
        """
        r = max(0.0, min(1.0, float(risk)))
        return cls(
            margin_floor=_lerp(0.10, 0.0, r),
            assassin_ceiling=_lerp(0.25, 0.45, r),
            ambition_weight=_lerp(0.04, 0.18, r),
            margin_weight=_lerp(0.50, 0.10, r),
            freq_weight=0.10,
            assassin_weight=_lerp(2.0, 0.5, r),
            opponent_weight=_lerp(0.30, 0.10, r),
            prefer_min_targets=3,
            undercluster_penalty_weight=_lerp(0.12, 0.06, r),
            expected_reward_weight=_lerp(1.10, 0.85, r),
            mc_trials=96,
            mc_temperature=_lerp(0.14, 0.22, r),
            reward_friendly=1.0,
            reward_neutral=-0.35,
            reward_opponent=-0.8,
            reward_assassin=-3.0,
        )


def freq_bonus(zipf: float, freq_weight: float) -> float:
    """Saturating bonus for common words; plateaus past Zipf ~5+."""
    return freq_weight * math.tanh((zipf - 3.0) / 2.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(frozen=True)
class StopPolicy:
    """Guesser stopping behaviour, derived from a single `risk` knob.

    `confidence_floor` controls picks 2..N: below the floor, stop short.
    A very negative floor disables stopping (always takes all N).

    `bonus_gap_threshold` controls the optional N+1 bonus pick: it's taken only
    when the gap between the Nth and (N+1)th similarity is *below* the
    threshold (i.e. the (N+1)th is comparably good). Negative threshold
    disables the bonus entirely (used at low risk).
    """

    confidence_floor: float
    bonus_gap_threshold: float
    risk: float

    @classmethod
    def from_risk(cls, risk: float) -> "StopPolicy":
        r = max(0.0, min(1.0, float(risk)))
        return cls(
            confidence_floor=_lerp(0.30, -1.0, r),
            bonus_gap_threshold=_lerp(-1.0, 0.20, r),
            risk=r,
        )
