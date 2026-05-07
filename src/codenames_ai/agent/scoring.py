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
    """Hard veto: reject candidates whose similarity to the assassin exceeds this."""

    ambition_weight: float
    """Bonus per additional target word in the subset (multiplied by N-1)."""

    margin_weight: float
    """Multiplier on margin for the soft scoring component."""

    freq_weight: float
    """Scale of the saturating frequency bonus."""

    assassin_weight: float
    """Penalty multiplier on similarity to the assassin (in addition to the hard veto)."""

    opponent_weight: float
    """Penalty multiplier on the worst (max) opponent-card similarity."""

    @classmethod
    def from_risk(cls, risk: float) -> "ScoringWeights":
        """Map a single `risk` scalar in [0, 1] to a weight set.

        risk = 0 → cautious: high margin requirements, low ambition, big assassin
        penalties. risk = 1 → aggressive: thin margins acceptable, ambition rewarded.
        """
        r = max(0.0, min(1.0, float(risk)))
        return cls(
            margin_floor=_lerp(0.10, 0.0, r),
            assassin_ceiling=_lerp(0.25, 0.45, r),
            ambition_weight=_lerp(0.02, 0.15, r),
            margin_weight=_lerp(0.50, 0.10, r),
            freq_weight=0.10,
            assassin_weight=_lerp(2.0, 0.5, r),
            opponent_weight=_lerp(0.30, 0.10, r),
        )


def freq_bonus(zipf: float, freq_weight: float) -> float:
    """Saturating bonus that promotes more-common words but plateaus past Zipf ~5+."""
    return freq_weight * math.tanh((zipf - 3.0) / 2.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(frozen=True)
class StopPolicy:
    """Guesser stopping behaviour, derived from a single `risk` knob.

    `confidence_floor` controls picks 2..N: if a candidate's similarity is below
    the floor, the guesser stops short rather than committing. Setting the
    floor very negative effectively disables stopping (always takes all N).

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
