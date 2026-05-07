import pytest

from codenames_ai.game.models import Board, Card, Clue, Color
from codenames_ai.game.state import (
    GameState,
    TurnEvent,
    TurnPhase,
    check_win,
    reveal_card,
)


def _board(cards: list[tuple[str, Color, bool]]) -> Board:
    full = list(cards)
    if len(full) > 25:
        raise ValueError("too many cards")
    full = full + [(f"pad{i}", Color.NEUTRAL, True) for i in range(25 - len(full))]
    return Board(
        cards=tuple(Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in full),
        first_team=Color.RED,
    )


def _state(board, history=()):
    return GameState(
        board=board,
        turn_history=tuple(history),
        current_team=Color.RED,
        current_phase=TurnPhase.SPYMASTER,
        winner=None,
        rng_seed=0,
    )


class TestRevealCard:
    def test_marks_card_revealed(self):
        board = _board([("apple", Color.RED, False)])
        new = reveal_card(board, "apple")
        assert next(c for c in new.cards if c.word == "apple").revealed is True

    def test_returns_new_board_unchanged_otherwise(self):
        board = _board([("apple", Color.RED, False), ("banana", Color.BLUE, False)])
        new = reveal_card(board, "apple")
        assert next(c for c in new.cards if c.word == "banana").revealed is False
        # Original is unchanged (frozen dataclass invariant).
        assert next(c for c in board.cards if c.word == "apple").revealed is False

    def test_raises_on_unknown_word(self):
        board = _board([("apple", Color.RED, False)])
        with pytest.raises(ValueError, match="not on board"):
            reveal_card(board, "missing")

    def test_raises_on_already_revealed(self):
        board = _board([("apple", Color.RED, True)])
        with pytest.raises(ValueError, match="already revealed"):
            reveal_card(board, "apple")


class TestCheckWin:
    def test_returns_none_for_active_game(self):
        board = _board([("a", Color.RED, False), ("b", Color.BLUE, False)])
        assert check_win(_state(board)) is None

    def test_red_wins_when_all_red_revealed(self):
        # 9 RED cards all revealed; rest unrevealed.
        cards = [(f"r{i}", Color.RED, True) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE, False) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL, False) for i in range(7)]
        cards += [("ass", Color.ASSASSIN, False)]
        board = Board(
            cards=tuple(
                Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in cards
            ),
            first_team=Color.RED,
        )
        assert check_win(_state(board)) == Color.RED

    def test_blue_wins_when_all_blue_revealed(self):
        cards = [(f"r{i}", Color.RED, False) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE, True) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL, False) for i in range(7)]
        cards += [("ass", Color.ASSASSIN, False)]
        board = Board(
            cards=tuple(
                Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in cards
            ),
            first_team=Color.RED,
        )
        assert check_win(_state(board)) == Color.BLUE

    def test_assassin_loses_for_revealing_team(self):
        # Red revealed the assassin → blue wins.
        board = _board([("ass", Color.ASSASSIN, True)])
        history = (
            TurnEvent(
                team=Color.RED,
                kind="GUESS",
                guess="ass",
                outcome_color=Color.ASSASSIN,
            ),
        )
        assert check_win(_state(board, history)) == Color.BLUE

    def test_assassin_blue_revealing_loses(self):
        board = _board([("ass", Color.ASSASSIN, True)])
        history = (
            TurnEvent(
                team=Color.BLUE,
                kind="GUESS",
                guess="ass",
                outcome_color=Color.ASSASSIN,
            ),
        )
        assert check_win(_state(board, history)) == Color.RED


class TestStateHelpers:
    def test_latest_clue_returns_most_recent(self):
        board = _board([])
        history = (
            TurnEvent(team=Color.RED, kind="CLUE", clue=Clue("apple", 2)),
            TurnEvent(
                team=Color.RED,
                kind="GUESS",
                guess="x",
                outcome_color=Color.RED,
            ),
            TurnEvent(team=Color.BLUE, kind="CLUE", clue=Clue("banana", 1)),
        )
        assert _state(board, history).latest_clue() == Clue("banana", 1)

    def test_latest_clue_is_none_with_empty_history(self):
        board = _board([])
        assert _state(board).latest_clue() is None

    def test_score_counts_revealed_per_team(self):
        cards = [(f"r{i}", Color.RED, i < 3) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE, i < 5) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL, False) for i in range(7)]
        cards += [("ass", Color.ASSASSIN, False)]
        board = Board(
            cards=tuple(
                Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in cards
            ),
            first_team=Color.RED,
        )
        assert _state(board).score() == {Color.RED: 3, Color.BLUE: 5}

    def test_cards_remaining(self):
        cards = [(f"r{i}", Color.RED, i < 3) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE, False) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL, False) for i in range(7)]
        cards += [("ass", Color.ASSASSIN, False)]
        board = Board(
            cards=tuple(
                Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in cards
            ),
            first_team=Color.RED,
        )
        state = _state(board)
        assert state.cards_remaining(Color.RED) == 6
        assert state.cards_remaining(Color.BLUE) == 8
