from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codenames_ai.agent.trace import GuesserTrace, SpymasterTrace
from codenames_ai.game.human import HumanGuesser, HumanSpymaster
from codenames_ai.game.models import Clue, Color
from codenames_ai.game.orchestrator import Game

Role = Literal["human", "ai"]


@dataclass
class PlaySession:
    id: str
    game: Game
    roles: dict[Color, dict[str, Role]]
    humans: dict[str, HumanSpymaster | HumanGuesser]
    risk: float
    #: Set after a human guess POST; consumed on next GET for one-shot UI feedback.
    ui_guess_flash: dict[str, str] | None = None
    #: Incremented on each live broadcast so clients can merge REST vs WS without stale ties.
    live_mutation_seq: int = 0
    #: Latest AI decision (one of the two); exposed in API snapshots when ``include_secret_colors`` is true.
    last_ai_spymaster: tuple[Color, SpymasterTrace] | None = None
    last_ai_guesser: tuple[Color, Clue, GuesserTrace] | None = None
