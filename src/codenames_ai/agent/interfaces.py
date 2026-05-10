from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from codenames_ai.agent.trace import GuesserTrace, SpymasterTrace
from codenames_ai.game.models import Clue, Color, GuesserView, SpymasterView

if TYPE_CHECKING:
    from codenames_ai.game.state import TurnEvent


class Spymaster(ABC):
    """Produces a clue + count for a given board view."""

    @abstractmethod
    def give_clue(self, view: SpymasterView) -> SpymasterTrace: ...


@dataclass(frozen=True)
class RevealOutcome:
    """Result of revealing one card during a guesser turn.

    Returned by the orchestrator's reveal callback to a per-step ``Guesser`` so
    it can decide whether to keep guessing. ``view`` is the post-reveal
    ``GuesserView`` (or ``None`` once the turn ended). ``turn_ended`` is True
    when the engine has stopped accepting further guesses for this clue (wrong
    color, attempts exhausted, or game over).
    """

    view: GuesserView | None
    turn_ended: bool
    outcome_color: Color
    game_over: bool


class Guesser(ABC):
    """Produces an ordered guess sequence for a given board view + clue."""

    @abstractmethod
    def guess(self, view: GuesserView, clue: Clue) -> GuesserTrace: ...

    def play_turn(
        self,
        view: GuesserView,
        clue: Clue,
        history: tuple["TurnEvent", ...],
        *,
        reveal: Callable[[str], RevealOutcome],
    ) -> GuesserTrace:
        """Drive one full guesser turn through ``reveal`` callbacks.

        Default implementation preserves legacy batch semantics: it calls
        ``self.guess(view, clue)`` once and applies each predicted word in
        order until the engine signals the turn has ended. LLM-primary guessers
        override this to issue one LLM call per physical guess with a fresh
        ``GuesserView`` between reveals.

        ``history`` is the full ``GameState.turn_history`` up to (and including)
        the current clue event. Implementations may consume it for compressed
        per-team turn timelines used in prompts.
        """
        del history  # default impl ignores history
        trace = self.guess(view, clue)
        for word in trace.guesses:
            outcome = reveal(word)
            if outcome.turn_ended:
                break
        return trace


class NoLegalClueError(RuntimeError):
    """Raised when no candidate clue passes the hard vetoes (and no relaxation is configured)."""
