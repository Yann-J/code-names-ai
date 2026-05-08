from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from codenames_ai.agent.interfaces import Guesser, Spymaster
from codenames_ai.game.board import generate_board
from codenames_ai.game.models import Board, Color
from codenames_ai.game.orchestrator import Game
from codenames_ai.game.state import GameState
from codenames_ai.vocab.models import Vocabulary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameRecord:
    """One game outcome and summary stats from `final_state`.

    Designed to round-trip through parquet without keeping the full board /
    history in row form (those live in `final_state` for in-memory inspection).
    """

    seed: int
    initial_board: Board
    final_state: GameState
    label: str = ""
    config_hash: str = ""

    @property
    def winner(self) -> Color | None:
        return self.final_state.winner

    @property
    def first_team(self) -> Color:
        return self.initial_board.first_team

    @property
    def num_clues(self) -> int:
        return sum(
            1 for ev in self.final_state.turn_history if ev.kind == "CLUE"
        )

    @property
    def num_guesses(self) -> int:
        return sum(
            1 for ev in self.final_state.turn_history if ev.kind == "GUESS"
        )

    @property
    def correct_guesses(self) -> int:
        """Guesses where outcome color matches the guessing team's color."""
        return sum(
            1
            for ev in self.final_state.turn_history
            if ev.kind == "GUESS" and ev.outcome_color == ev.team
        )

    @property
    def assassin_hit(self) -> bool:
        return any(
            ev.kind == "GUESS" and ev.outcome_color == Color.ASSASSIN
            for ev in self.final_state.turn_history
        )

    @property
    def clue_counts(self) -> tuple[int, ...]:
        return tuple(
            int(ev.clue.count)
            for ev in self.final_state.turn_history
            if ev.kind == "CLUE" and ev.clue is not None
        )

    @property
    def avg_clue_count(self) -> float:
        counts = self.clue_counts
        if not counts:
            return 0.0
        return sum(counts) / len(counts)

    @property
    def clue_rate_ge_2(self) -> float:
        counts = self.clue_counts
        if not counts:
            return 0.0
        return sum(1 for c in counts if c >= 2) / len(counts)

    @property
    def clue_rate_ge_3(self) -> float:
        counts = self.clue_counts
        if not counts:
            return 0.0
        return sum(1 for c in counts if c >= 3) / len(counts)


def run_tournament(
    *,
    seeds: Iterable[int],
    game_vocab: Vocabulary,
    red_spymaster: Spymaster,
    red_guesser: Guesser,
    blue_spymaster: Spymaster,
    blue_guesser: Guesser,
    max_clues: int = 30,
    label: str = "",
    config_hash: str = "",
) -> list[GameRecord]:
    """Run a sequence of full self-play games.

    Players are reused across games — they're expected to be stateless between
    `give_clue` / `guess` calls, which both `AISpymaster` and `AIGuesser` are.
    """
    records: list[GameRecord] = []
    for seed in seeds:
        board = generate_board(game_vocab, seed=seed)
        game = Game(
            board,
            red_spymaster=red_spymaster,
            red_guesser=red_guesser,
            blue_spymaster=blue_spymaster,
            blue_guesser=blue_guesser,
            seed=seed,
            max_clues=max_clues,
        )
        final = game.play()
        records.append(
            GameRecord(
                seed=seed,
                initial_board=board,
                final_state=final,
                label=label,
                config_hash=config_hash,
            )
        )
        logger.info(
            "tournament game seed=%d winner=%s clues=%d guesses=%d",
            seed,
            final.winner.value if final.winner else "none",
            sum(1 for ev in final.turn_history if ev.kind == "CLUE"),
            sum(1 for ev in final.turn_history if ev.kind == "GUESS"),
        )
    return records
