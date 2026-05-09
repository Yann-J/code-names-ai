from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

from codenames_ai.game.models import Board, Card, Clue, Color


class TurnPhase(str, Enum):
    SPYMASTER = "SPYMASTER"  # current team's spymaster is about to give a clue
    GUESSER = "GUESSER"  # current team's guesser is about to play
    DONE = "DONE"  # game over


@dataclass(frozen=True)
class TurnEvent:
    """One atomic event in the game's history.

    A clue event populates `clue`; a guess event populates `guess` and
    `outcome_color` (the color of the card that was revealed).
    """

    team: Color
    kind: Literal["CLUE", "GUESS"]
    clue: Clue | None = None
    guess: str | None = None
    outcome_color: Color | None = None


def prior_clue_surfaces_lower(
    turn_history: tuple[TurnEvent, ...],
) -> frozenset[str]:
    """Lowercased clue surfaces already played (Codenames forbids repeating a clue)."""
    return frozenset(
        ev.clue.word.lower()
        for ev in turn_history
        if ev.kind == "CLUE"
        and ev.clue is not None
        and not ev.clue.is_pass()
    )


@dataclass(frozen=True)
class GameState:
    board: Board
    turn_history: tuple[TurnEvent, ...]
    current_team: Color
    current_phase: TurnPhase
    winner: Color | None
    rng_seed: int
    #: Guesses left this clue (inclusive of the next pick); set in GUESSER phase.
    guesser_attempts_remaining: int | None = None

    @property
    def is_over(self) -> bool:
        return self.current_phase == TurnPhase.DONE

    def latest_clue(self) -> Clue | None:
        for ev in reversed(self.turn_history):
            if ev.kind == "CLUE":
                return ev.clue
        return None

    def guess_count_after_latest_clue(self) -> int:
        """Number of GUESS events since the most recent CLUE event (this clue round)."""
        last_clue_idx = -1
        for i, ev in enumerate(self.turn_history):
            if ev.kind == "CLUE":
                last_clue_idx = i
        if last_clue_idx < 0:
            return 0
        return sum(
            1 for ev in self.turn_history[last_clue_idx + 1 :] if ev.kind == "GUESS"
        )

    def score(self) -> dict[Color, int]:
        """Revealed-card count per team."""
        return {
            Color.RED: sum(
                1 for c in self.board.cards if c.color == Color.RED and c.revealed
            ),
            Color.BLUE: sum(
                1 for c in self.board.cards if c.color == Color.BLUE and c.revealed
            ),
        }

    def cards_remaining(self, team: Color) -> int:
        """Unrevealed cards for `team`."""
        return sum(
            1 for c in self.board.cards if c.color == team and not c.revealed
        )


def reveal_card(board: Board, word: str) -> Board:
    """Return a new board with the matching card flipped to `revealed=True`.

    Raises if the word isn't on the board or is already revealed.
    """
    new_cards: list[Card] = []
    found = False
    for c in board.cards:
        if c.word == word and not c.revealed:
            new_cards.append(replace(c, revealed=True))
            found = True
        else:
            new_cards.append(c)
    if not found:
        raise ValueError(f"card {word!r} not on board or already revealed")
    return replace(board, cards=tuple(new_cards))


def check_win(state: GameState) -> Color | None:
    """Return the winning team if the game is over, else None.

    Two end conditions:
      1. A team revealed the assassin → that team's *opponent* wins.
      2. A team revealed all of their own cards → they win.

    Assassin takes precedence (and is detectable from the most recent guess
    event in the history).
    """
    for ev in reversed(state.turn_history):
        if ev.kind == "GUESS" and ev.outcome_color == Color.ASSASSIN:
            return ev.team.opponent()
        # Only need to check the latest guess: assassin reveal is terminal.
        if ev.kind == "GUESS":
            break

    for team in (Color.RED, Color.BLUE):
        total = sum(1 for c in state.board.cards if c.color == team)
        revealed = sum(
            1 for c in state.board.cards if c.color == team and c.revealed
        )
        if total > 0 and revealed == total:
            return team
    return None
