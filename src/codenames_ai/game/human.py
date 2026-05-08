from __future__ import annotations

from codenames_ai.agent.interfaces import Guesser, Spymaster
from codenames_ai.agent.scoring import ScoringWeights, StopPolicy
from codenames_ai.agent.trace import Candidate, GuesserTrace, ScoreComponents, SpymasterTrace
from codenames_ai.game.models import Clue, GuesserView, SpymasterView


def trivial_spymaster_trace(
    clue: str,
    *,
    targets: tuple[str, ...],
    n: int,
    weights: ScoringWeights | None = None,
) -> SpymasterTrace:
    """Build a minimal trace for a human-chosen legal clue (notebook / web hook-in)."""
    w = weights or ScoringWeights.from_risk(0.5)
    z = ScoreComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cand = Candidate(
        clue=clue,
        targets=targets,
        n=n,
        score=0.0,
        embedding_score=0.0,
        components=z,
        margin=0.0,
        zipf=5.0,
    )
    return SpymasterTrace(
        chosen=cand,
        top_candidates=(cand,),
        weights=w,
        veto_count=0,
        illegal_count=0,
    )


def trivial_guesser_trace(
    guesses: tuple[str, ...],
    *,
    stop_policy: StopPolicy | None = None,
    stop_reason: str = "human",
) -> GuesserTrace:
    return GuesserTrace(
        candidates=(),
        guesses=guesses,
        stop_policy=stop_policy or StopPolicy.from_risk(0.5),
        bonus_attempted=False,
        stop_reason=stop_reason,
    )


class HumanSpymaster(Spymaster):
    """Supply the next clue via `prepare()` before `Game.step()` reaches this player."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: SpymasterTrace | None = None

    def prepare(self, trace: SpymasterTrace) -> None:
        if self._pending is not None:
            raise RuntimeError("HumanSpymaster: prepare() called twice without give_clue()")
        self._pending = trace

    def give_clue(self, view: SpymasterView) -> SpymasterTrace:
        if self._pending is None:
            raise RuntimeError(
                "HumanSpymaster: call prepare(trace) on the web/notebook layer before stepping"
            )
        t = self._pending
        self._pending = None
        return t


class HumanGuesser(Guesser):
    """Supply guesses via `prepare()` before the guesser phase runs."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: GuesserTrace | None = None

    def prepare(self, trace: GuesserTrace) -> None:
        if self._pending is not None:
            raise RuntimeError("HumanGuesser: prepare() called twice without guess()")
        self._pending = trace

    def guess(self, view: GuesserView, clue: Clue) -> GuesserTrace:
        if self._pending is None:
            raise RuntimeError(
                "HumanGuesser: call prepare(trace) before stepping the guesser phase"
            )
        t = self._pending
        self._pending = None
        return t
