from __future__ import annotations

from abc import ABC, abstractmethod

from codenames_ai.agent.trace import GuesserTrace, SpymasterTrace
from codenames_ai.game.models import Clue, GuesserView, SpymasterView


class Spymaster(ABC):
    """Produces a clue + count for a given board view."""

    @abstractmethod
    def give_clue(self, view: SpymasterView) -> SpymasterTrace: ...


class Guesser(ABC):
    """Produces an ordered guess sequence for a given board view + clue."""

    @abstractmethod
    def guess(self, view: GuesserView, clue: Clue) -> GuesserTrace: ...


class NoLegalClueError(RuntimeError):
    """Raised when no candidate clue passes the hard vetoes (and no relaxation is configured)."""
