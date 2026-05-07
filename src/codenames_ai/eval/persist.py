from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from codenames_ai.eval.tournament import GameRecord


_RECORD_COLUMNS = [
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


def records_to_dataframe(records: Sequence[GameRecord]) -> pd.DataFrame:
    """One row per game with summary columns. Board/history are NOT serialized."""
    rows = []
    for r in records:
        rows.append(
            {
                "label": r.label,
                "config_hash": r.config_hash,
                "seed": r.seed,
                "winner": r.winner.value if r.winner else None,
                "first_team": r.first_team.value,
                "num_clues": r.num_clues,
                "num_guesses": r.num_guesses,
                "correct_guesses": r.correct_guesses,
                "assassin_hit": r.assassin_hit,
            }
        )
    return pd.DataFrame(rows, columns=_RECORD_COLUMNS)


def save_records(records: Sequence[GameRecord], path: Path) -> None:
    """Persist tournament records to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    df.to_parquet(path, index=False)


def load_records_dataframe(path: Path) -> pd.DataFrame:
    """Read previously saved tournament records back as a DataFrame."""
    return pd.read_parquet(path)
