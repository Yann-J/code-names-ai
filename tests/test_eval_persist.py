import pandas as pd

from codenames_ai.eval.persist import (
    load_records_dataframe,
    records_to_dataframe,
    save_records,
)
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


def _record(*, winner: Color | None = Color.RED, label: str = "v1") -> GameRecord:
    history = (
        TurnEvent(team=Color.RED, kind="GUESS", guess="a", outcome_color=Color.RED),
    )
    state = GameState(
        board=_empty_board(),
        turn_history=history,
        current_team=Color.RED,
        current_phase=TurnPhase.DONE,
        winner=winner,
        rng_seed=0,
    )
    return GameRecord(
        seed=42,
        initial_board=_empty_board(),
        final_state=state,
        label=label,
    )


class TestRecordsToDataFrame:
    def test_columns(self):
        df = records_to_dataframe([_record()])
        assert list(df.columns) == [
            "label",
            "config_hash",
            "seed",
            "winner",
            "first_team",
            "num_clues",
            "num_guesses",
            "correct_guesses",
            "assassin_hit",
        ]

    def test_winner_serialized_as_string(self):
        df = records_to_dataframe([_record(winner=Color.BLUE)])
        assert df.iloc[0]["winner"] == "BLUE"

    def test_no_winner_is_none(self):
        df = records_to_dataframe([_record(winner=None)])
        assert df.iloc[0]["winner"] is None


class TestRoundTrip:
    def test_save_then_load(self, tmp_path):
        records = [_record(winner=Color.RED, label="exp1"), _record(winner=Color.BLUE)]
        path = tmp_path / "out.parquet"
        save_records(records, path)

        loaded = load_records_dataframe(path)
        expected = records_to_dataframe(records)
        pd.testing.assert_frame_equal(loaded, expected)

    def test_save_creates_parent_directory(self, tmp_path):
        records = [_record()]
        path = tmp_path / "nested" / "deep" / "out.parquet"
        save_records(records, path)
        assert path.exists()
