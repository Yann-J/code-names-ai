from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Color(str, Enum):
    RED = "RED"
    BLUE = "BLUE"
    NEUTRAL = "NEUTRAL"
    ASSASSIN = "ASSASSIN"

    @property
    def is_team(self) -> bool:
        return self in (Color.RED, Color.BLUE)

    def opponent(self) -> "Color":
        if self is Color.RED:
            return Color.BLUE
        if self is Color.BLUE:
            return Color.RED
        raise ValueError(f"{self} has no opponent")


@dataclass(frozen=True)
class Card:
    word: str  # surface form, lowercased
    lemma: str
    color: Color
    revealed: bool = False


@dataclass(frozen=True)
class Board:
    cards: tuple[Card, ...]
    first_team: Color  # the team holding 9 cards (the other holds 8)

    def __post_init__(self) -> None:
        if len(self.cards) != 25:
            raise ValueError(f"Board must have 25 cards, got {len(self.cards)}")
        if not self.first_team.is_team:
            raise ValueError(f"first_team must be RED or BLUE, got {self.first_team}")

    def active(self) -> tuple[Card, ...]:
        return tuple(c for c in self.cards if not c.revealed)

    def cards_of(self, color: Color) -> tuple[Card, ...]:
        return tuple(c for c in self.cards if c.color == color)


@dataclass(frozen=True)
class Clue:
    word: str
    count: int  # the N declared by the spymaster

    def is_pass(self) -> bool:
        return self.word == "" and self.count == 0


@dataclass(frozen=True)
class SpymasterView:
    """A spymaster's view of the game: full board (all colors visible to them).

    ``prior_clue_words`` collects normalized clue surfaces already played this
    match (Codenames forbids repeating a clue word). Static vocabulary exclusions
    are applied separately inside ``AISpymaster``.
    """

    board: Board
    team: Color
    prior_clue_words: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class GuesserView:
    """A guesser's view: same board but colors only revealed for played cards.

    Placeholder for M4; M3 only needs the SpymasterView.
    """

    board: Board
    team: Color
