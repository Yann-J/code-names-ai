from __future__ import annotations

from collections.abc import Sequence

from codenames_ai.eval.tournament import GameRecord
from codenames_ai.game.models import Color


def aggregate(records: Sequence[GameRecord]) -> dict[str, float]:
    """Summary stats across a tournament's games.

    Keys returned (always present, possibly 0 for empty inputs):
      `n_games`, `red_win_rate`, `blue_win_rate`, `no_winner_rate`,
      `assassin_rate`, `avg_clues`, `avg_guesses`, `avg_guesses_per_clue`,
      `accuracy` (own-color / total-guess).
    """
    n = len(records)
    if n == 0:
        return {
            "n_games": 0,
            "red_win_rate": 0.0,
            "blue_win_rate": 0.0,
            "no_winner_rate": 0.0,
            "assassin_rate": 0.0,
            "avg_clues": 0.0,
            "avg_guesses": 0.0,
            "avg_guesses_per_clue": 0.0,
            "accuracy": 0.0,
        }

    total_clues = sum(r.num_clues for r in records)
    total_guesses = sum(r.num_guesses for r in records)
    total_correct = sum(r.correct_guesses for r in records)

    return {
        "n_games": n,
        "red_win_rate": sum(1 for r in records if r.winner == Color.RED) / n,
        "blue_win_rate": sum(1 for r in records if r.winner == Color.BLUE) / n,
        "no_winner_rate": sum(1 for r in records if r.winner is None) / n,
        "assassin_rate": sum(1 for r in records if r.assassin_hit) / n,
        "avg_clues": total_clues / n,
        "avg_guesses": total_guesses / n,
        "avg_guesses_per_clue": (total_guesses / total_clues) if total_clues else 0.0,
        "accuracy": (total_correct / total_guesses) if total_guesses else 0.0,
    }


def compare(label_to_records: dict[str, Sequence[GameRecord]]) -> list[dict[str, float]]:
    """Tabular form: one row per label, each row = aggregate(records) + {label}.

    Convenient for `pd.DataFrame(compare(...))` in a notebook.
    """
    rows: list[dict[str, float]] = []
    for label, records in label_to_records.items():
        row = {"label": label}
        row.update(aggregate(records))
        rows.append(row)
    return rows
