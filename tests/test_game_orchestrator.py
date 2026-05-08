"""Orchestrator tests using scripted Spymaster/Guesser fakes.

Each test composes a full game from synthetic players whose responses are
canned, so the game-flow logic is exercised without depending on real
embeddings or LLMs.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import pytest

from codenames_ai.agent.interfaces import Guesser, NoLegalClueError, Spymaster
from codenames_ai.agent.scoring import ScoringWeights, StopPolicy
from codenames_ai.agent.trace import (
    Candidate,
    CandidateGuess,
    GuesserTrace,
    ScoreComponents,
    SpymasterTrace,
)
from codenames_ai.game.models import Board, Card, Clue, Color
from codenames_ai.game.orchestrator import Game
from codenames_ai.game.state import TurnPhase


def _candidate(clue: str, n: int) -> Candidate:
    components = ScoreComponents(
        friendly_min_sim=1.0,
        ambition_bonus=0.0,
        margin_bonus=0.0,
        freq_bonus=0.0,
        assassin_penalty=0.0,
        opponent_penalty=0.0,
        expected_reward_bonus=0.0,
        expected_reward_raw=0.0,
    )
    return Candidate(
        clue=clue,
        targets=tuple(),
        n=n,
        score=1.0,
        embedding_score=1.0,
        components=components,
        margin=1.0,
        zipf=5.0,
    )


def _spy_trace(clue: str, n: int) -> SpymasterTrace:
    cand = _candidate(clue, n)
    return SpymasterTrace(
        chosen=cand,
        top_candidates=(cand,),
        weights=ScoringWeights.from_risk(0.5),
        veto_count=0,
        illegal_count=0,
    )


def _guess_trace(words: list[str]) -> GuesserTrace:
    cands = tuple(
        CandidateGuess(
            word=w,
            similarity=1.0,
            score=1.0,
            rank=i,
            committed=True,
            is_bonus=False,
        )
        for i, w in enumerate(words)
    )
    return GuesserTrace(
        candidates=cands,
        guesses=tuple(words),
        stop_policy=StopPolicy.from_risk(0.5),
        bonus_attempted=False,
        stop_reason="reached_n",
    )


class ScriptedSpymaster(Spymaster):
    def __init__(self, clues: Sequence[tuple[str, int]]):
        self._clues = deque(clues)
        self.calls = 0

    def give_clue(self, view):
        self.calls += 1
        if not self._clues:
            raise NoLegalClueError("scripted spymaster ran out of clues")
        word, n = self._clues.popleft()
        return _spy_trace(word, n)


class ScriptedGuesser(Guesser):
    def __init__(self, guesses: Sequence[Sequence[str]]):
        self._guesses = deque(list(g) for g in guesses)
        self.calls = 0

    def guess(self, view, clue):
        self.calls += 1
        words = list(self._guesses.popleft()) if self._guesses else []
        return _guess_trace(words)


def _board(cards: list[tuple[str, Color]]) -> Board:
    if len(cards) > 25:
        raise ValueError("too many")
    full = list(cards) + [
        (f"pad{i}", Color.NEUTRAL) for i in range(25 - len(cards))
    ]
    return Board(
        cards=tuple(Card(word=w, lemma=w, color=c) for w, c in full),
        first_team=Color.RED,
    )


def _make_game(
    *,
    board: Board,
    red_clues: Sequence[tuple[str, int]] = (("clue", 1),),
    red_guesses: Sequence[Sequence[str]] = ((),),
    blue_clues: Sequence[tuple[str, int]] = (("clue", 1),),
    blue_guesses: Sequence[Sequence[str]] = ((),),
    max_clues: int = 50,
) -> Game:
    return Game(
        board,
        red_spymaster=ScriptedSpymaster(red_clues),
        red_guesser=ScriptedGuesser(red_guesses),
        blue_spymaster=ScriptedSpymaster(blue_clues),
        blue_guesser=ScriptedGuesser(blue_guesses),
        max_clues=max_clues,
    )


class TestStateMachine:
    def test_starts_with_first_team_spymaster_phase(self):
        board = _board(
            [("r0", Color.RED), ("b0", Color.BLUE), ("ass", Color.ASSASSIN)]
        )
        game = _make_game(board=board)
        assert game.state.current_team == Color.RED
        assert game.state.current_phase == TurnPhase.SPYMASTER

    def test_step_advances_through_phases(self):
        # Need >1 RED card so revealing one doesn't immediately end the game.
        cards = [("r0", Color.RED), ("r1", Color.RED), ("b0", Color.BLUE)]
        cards += [(f"r{i}", Color.RED) for i in range(2, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(1, 8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("c", 1)],
            red_guesses=[["r0"]],
            blue_clues=[("c", 1)],
            blue_guesses=[["b0"]],
        )
        # First step: spymaster gives a clue.
        game.step()
        assert game.state.current_phase == TurnPhase.GUESSER
        assert len(game.spymaster_traces) == 1

        # Second step: guesser plays — turn ends, switch to BLUE spymaster.
        game.step()
        assert game.state.current_team == Color.BLUE
        assert game.state.current_phase == TurnPhase.SPYMASTER


class TestRedTeamWinPath:
    def test_red_wins_by_revealing_all_red_cards(self):
        # Game is rigged: only RED cards are unrevealed; one clue reveals them all.
        red_words = [f"r{i}" for i in range(9)]
        cards = [(w, Color.RED) for w in red_words]
        cards += [(f"b{i}", Color.BLUE) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("everything", 9)],
            red_guesses=[red_words],
        )
        final = game.play()
        assert final.is_over
        assert final.winner == Color.RED


class TestAssassinPath:
    def test_revealing_assassin_loses_immediately(self):
        cards = [(f"r{i}", Color.RED) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("c", 1)],
            red_guesses=[["ass"]],
        )
        final = game.play()
        assert final.is_over
        assert final.winner == Color.BLUE  # red revealed assassin → blue wins


class TestWrongColorEndsTurn:
    def test_wrong_color_terminates_remaining_picks(self):
        cards = [("r0", Color.RED), ("r1", Color.RED), ("b0", Color.BLUE)]
        cards += [(f"r{i}", Color.RED) for i in range(2, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(1, 8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        # Red would-be-pick: r0 (correct), b0 (wrong, ends turn), r1 (skipped).
        game = _make_game(
            board=board,
            red_clues=[("c", 3)],
            red_guesses=[["r0", "b0", "r1"]],
            blue_clues=[("c", 0)],
            blue_guesses=[[]],
        )
        # Step through one full red turn.
        game.step()  # red spymaster
        game.step()  # red guesser

        # r0 revealed, b0 revealed, r1 NOT revealed.
        revealed = {c.word for c in game.state.board.cards if c.revealed}
        assert "r0" in revealed
        assert "b0" in revealed
        assert "r1" not in revealed


class TestDeterminism:
    def test_same_seed_same_final_state(self):
        cards = [(f"r{i}", Color.RED) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )

        def run() -> tuple:
            game = _make_game(
                board=board,
                red_clues=[("c", 9)],
                red_guesses=[[f"r{i}" for i in range(9)]],
            )
            return game.play()

        a = run()
        b = run()
        assert a.winner == b.winner
        assert a.turn_history == b.turn_history


class TestPassHandling:
    def test_spymaster_pass_advances_to_opponent(self):
        cards = [(f"r{i}", Color.RED) for i in range(9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[],  # red spymaster will raise NoLegalClueError
            blue_clues=[("everything", 8)],
            blue_guesses=[[f"b{i}" for i in range(8)]],
        )
        final = game.play()
        # Red passed; blue then revealed all 8 cards and wins.
        assert final.winner == Color.BLUE


class TestMaxClueCap:
    def test_terminates_after_max_clues_without_winner(self):
        board = _board(
            [("r0", Color.RED), ("b0", Color.BLUE), ("ass", Color.ASSASSIN)]
        )
        # Both teams give clues that don't progress (empty guesses).
        game = _make_game(
            board=board,
            red_clues=[("c", 0)] * 100,
            red_guesses=[[]] * 100,
            blue_clues=[("c", 0)] * 100,
            blue_guesses=[[]] * 100,
            max_clues=4,
        )
        final = game.play()
        assert final.is_over
        # No winner reached.
        assert final.winner is None


class TestHumanIncrementalGuess:
    """Web UI submits one guess at a time via ``apply_human_guess``."""

    def test_single_correct_guess_stays_in_guesser_until_cap(self):
        cards = [("r0", Color.RED), ("r1", Color.RED), ("b0", Color.BLUE)]
        cards += [(f"r{i}", Color.RED) for i in range(2, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(1, 8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("c", 2)],
            red_guesses=[[]],
            blue_clues=[("c", 1)],
            blue_guesses=[["b0"]],
        )
        game.step()
        assert game.state.current_phase == TurnPhase.GUESSER
        assert game.state.guesser_attempts_remaining == 3

        game.apply_human_guess("r0")
        assert game.state.current_phase == TurnPhase.GUESSER
        assert game.state.current_team == Color.RED
        assert game.state.guesser_attempts_remaining == 2

        game.apply_human_guess("r1")
        assert game.state.current_phase == TurnPhase.GUESSER
        assert game.state.current_team == Color.RED
        assert game.state.guesser_attempts_remaining == 1

        game.apply_human_guess("r2")
        assert game.state.current_phase == TurnPhase.SPYMASTER
        assert game.state.current_team == Color.BLUE

    def test_end_guessing_turn_after_one_correct(self):
        cards = [("r0", Color.RED), ("r1", Color.RED), ("b0", Color.BLUE)]
        cards += [(f"r{i}", Color.RED) for i in range(2, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(1, 8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("c", 3)],
            red_guesses=[[]],
            blue_clues=[("c", 1)],
            blue_guesses=[["b0"]],
        )
        game.step()
        game.apply_human_guess("r0")
        assert game.state.guess_count_after_latest_clue() == 1
        game.end_guessing_turn()
        assert game.state.current_phase == TurnPhase.SPYMASTER
        assert game.state.current_team == Color.BLUE

    def test_end_guessing_turn_requires_at_least_one_guess(self):
        cards = [("r0", Color.RED)]
        cards += [(f"r{i}", Color.RED) for i in range(1, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("c", 1)],
            red_guesses=[[]],
            blue_clues=[("c", 1)],
            blue_guesses=[["b0"]],
        )
        game.step()
        assert game.state.guess_count_after_latest_clue() == 0
        with pytest.raises(ValueError, match="at least one guess"):
            game.end_guessing_turn()
        """Double-submit the same card would mark it revealed twice — ignore safely."""
        cards = [("r0", Color.RED), ("r1", Color.RED), ("b0", Color.BLUE)]
        cards += [(f"r{i}", Color.RED) for i in range(2, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(1, 8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        board = Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )
        game = _make_game(
            board=board,
            red_clues=[("c", 3)],
            red_guesses=[[]],
            blue_clues=[("c", 1)],
            blue_guesses=[["b0"]],
        )
        game.step()
        game.apply_human_guess("r0")
        assert game.state.current_phase == TurnPhase.GUESSER
        team_before = game.state.current_team
        game.apply_human_guess("r0")
        assert game.state.current_phase == TurnPhase.GUESSER
        assert game.state.current_team == team_before
