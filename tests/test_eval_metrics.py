from dataclasses import replace

from codenames_ai.eval.metrics import aggregate, compare
from codenames_ai.eval.tournament import GameRecord
from codenames_ai.game.models import Board, Card, Color
from codenames_ai.game.state import GameState, TurnEvent, TurnPhase


def _empty_board() -> Board:
    return Board(
        cards=tuple(
            Card(word=f"x{i}", lemma=f"x{i}", color=Color.NEUTRAL)
            for i in range(25)
        ),
        first_team=Color.RED,
    )


def _state(history: tuple[TurnEvent, ...], winner: Color | None = None) -> GameState:
    return GameState(
        board=_empty_board(),
        turn_history=history,
        current_team=Color.RED,
        current_phase=TurnPhase.DONE,
        winner=winner,
        rng_seed=0,
    )


def _record(
    *,
    seed: int = 0,
    history: tuple[TurnEvent, ...] = (),
    winner: Color | None = None,
    label: str = "",
) -> GameRecord:
    return GameRecord(
        seed=seed,
        initial_board=_empty_board(),
        final_state=_state(history, winner),
        label=label,
    )


class TestAggregateEmpty:
    def test_returns_zero_keys_for_empty(self):
        out = aggregate([])
        assert out["n_games"] == 0
        assert out["red_win_rate"] == 0.0
        assert out["accuracy"] == 0.0


class TestAggregateBasic:
    def test_red_win_rate(self):
        records = [
            _record(winner=Color.RED),
            _record(winner=Color.RED),
            _record(winner=Color.BLUE),
            _record(winner=None),
        ]
        out = aggregate(records)
        assert out["n_games"] == 4
        assert out["red_win_rate"] == 0.5
        assert out["blue_win_rate"] == 0.25
        assert out["no_winner_rate"] == 0.25

    def test_assassin_rate(self):
        history_normal = (
            TurnEvent(team=Color.RED, kind="GUESS", guess="x", outcome_color=Color.RED),
        )
        history_assassin = (
            TurnEvent(
                team=Color.RED,
                kind="GUESS",
                guess="x",
                outcome_color=Color.ASSASSIN,
            ),
        )
        records = [
            _record(history=history_normal),
            _record(history=history_assassin),
            _record(history=history_assassin),
        ]
        out = aggregate(records)
        assert out["assassin_rate"] == 2 / 3

    def test_accuracy_and_avg_guesses_per_clue(self):
        from codenames_ai.game.models import Clue

        history = (
            TurnEvent(team=Color.RED, kind="CLUE", clue=Clue("c", 3)),
            TurnEvent(
                team=Color.RED, kind="GUESS", guess="a", outcome_color=Color.RED
            ),
            TurnEvent(
                team=Color.RED, kind="GUESS", guess="b", outcome_color=Color.RED
            ),
            TurnEvent(
                team=Color.RED, kind="GUESS", guess="c", outcome_color=Color.NEUTRAL
            ),
        )
        records = [_record(history=history)]
        out = aggregate(records)
        assert out["avg_clues"] == 1.0
        assert out["avg_guesses"] == 3.0
        assert out["avg_guesses_per_clue"] == 3.0
        assert out["accuracy"] == 2 / 3


class TestCompare:
    def test_one_row_per_label(self):
        a = [_record(winner=Color.RED, label="A"), _record(winner=Color.BLUE, label="A")]
        b = [_record(winner=Color.RED, label="B")]
        rows = compare({"A": a, "B": b})
        labels = sorted(r["label"] for r in rows)
        assert labels == ["A", "B"]
