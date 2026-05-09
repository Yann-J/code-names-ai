from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Tunable weights for the spymaster: hard vetoes, MC EV rollout, and generation cap."""

    margin_floor: float
    """Hard veto: reject candidates whose margin is below this."""

    assassin_ceiling: float
    """Hard veto: reject if assassin similarity exceeds this."""

    mc_trials: int
    """Number of Monte Carlo rollouts used per `(clue, N)` candidate."""

    mc_temperature: float
    """Divides similarity logits before softmax; lower → sharper preference for high cosine."""

    mc_rank_bias: float
    """Subtracts ``bias * log1p(rank)`` from logits by descending similarity rank (0 = top).

    Use with stochastic MC: keeps sampling (unlike greedy argmax) but concentrates mass
    toward high-similarity cards when raw cosine gaps are small. Zero disables."""

    reward_friendly: float
    reward_neutral: float
    reward_opponent: float
    reward_assassin: float
    """Per-pick rewards used by Monte Carlo simulation."""

    lane_max_n: int
    """Maximum clue target count ``N`` evaluated (caps prefix size). Larger ``N`` is ignored."""

    adaptive_mc_base_trials: int
    """Base Monte Carlo trials before adaptive refinement."""

    adaptive_mc_extra_trials: int
    """Extra Monte Carlo trials for candidates near global EV leaders."""

    adaptive_mc_ev_band: float
    """EV distance to best surviving candidate that triggers extra trials."""

    @classmethod
    def from_risk(cls, risk: float) -> "ScoringWeights":
        """Map a single `risk` scalar in [0, 1] to vetoes and MC sampling sharpness.

        risk = 0 → cautious: stricter vetoes, lower ``mc_temperature`` (sharper cosines),
        higher ``mc_rank_bias`` (stochastic but greedy-leaning). risk = 1 is the converse.
        """
        r = max(0.0, min(1.0, float(risk)))
        return cls(
            margin_floor=_lerp(0.10, 0.0, r),
            assassin_ceiling=_lerp(0.25, 0.45, r),
            mc_trials=96,
            mc_temperature=_lerp(0.14, 0.22, r),
            mc_rank_bias=_lerp(1.6, 1.0, r),
            reward_friendly=1.0,
            reward_neutral=-0.35,
            reward_opponent=-0.8,
            reward_assassin=-3.0,
            lane_max_n=7,
            adaptive_mc_base_trials=64,
            adaptive_mc_extra_trials=96,
            adaptive_mc_ev_band=0.10,
        )


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
