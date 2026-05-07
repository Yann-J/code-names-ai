from __future__ import annotations

import numpy as np
import pytest

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.scoring import StopPolicy
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Board, Card, Clue, Color, GuesserView


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v


def make_matrix(entries: list[tuple[str, list[float]]]) -> EmbeddingMatrix:
    surfaces = [s for s, _ in entries]
    vectors = np.stack([_normalize(v) for _, v in entries])
    return EmbeddingMatrix(
        vectors=vectors,
        surfaces=surfaces,
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id="test",
        vocab_cache_key="test",
    )


def make_board(cards: list[tuple[str, Color, bool]]) -> Board:
    """Build a Board from `(word, color, revealed)` tuples, padding to 25 if short.

    Cards beyond what's specified are filled with placeholder NEUTRAL cards
    using surfaces 'pad0', 'pad1', etc. Each test ensures all referenced
    surfaces (including the padding ones) live in the matrix.
    """
    if len(cards) > 25:
        raise ValueError("too many cards")
    full = list(cards) + [(f"pad{i}", Color.NEUTRAL, True) for i in range(25 - len(cards))]
    # Ensure exactly 9 friendly (RED), 8 BLUE, 7 NEUTRAL, 1 ASSASSIN.
    # The Board invariant requires 25 cards; tests just need legal coverage,
    # so we don't enforce per-color counts — enforce only count == 25.
    if len(full) != 25:
        raise AssertionError("board not 25 cards")
    return Board(
        cards=tuple(Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in full),
        first_team=Color.RED,
    )


def _padded_matrix(extra_entries: list[tuple[str, list[float]]]) -> EmbeddingMatrix:
    """Matrix that includes 25 pad surfaces (any direction) plus the test's entries."""
    pad = [(f"pad{i}", [0.0, 0.0, 1.0]) for i in range(25)]
    return make_matrix(pad + extra_entries)


class TestStopPolicyFromRisk:
    def test_clamped(self):
        assert StopPolicy.from_risk(-1.0) == StopPolicy.from_risk(0.0)
        assert StopPolicy.from_risk(2.0) == StopPolicy.from_risk(1.0)

    def test_cautious_has_higher_confidence_floor(self):
        cautious = StopPolicy.from_risk(0.0)
        aggressive = StopPolicy.from_risk(1.0)
        assert cautious.confidence_floor > aggressive.confidence_floor

    def test_cautious_disables_bonus_via_negative_threshold(self):
        cautious = StopPolicy.from_risk(0.0)
        # gap is always >= 0; negative threshold means bonus is never taken.
        assert cautious.bonus_gap_threshold < 0

    def test_aggressive_enables_bonus(self):
        aggressive = StopPolicy.from_risk(1.0)
        assert aggressive.bonus_gap_threshold > 0


class TestBasicGuessing:
    def test_picks_top_n_when_all_above_floor(self):
        # Clue 'fruit' aligned with 'apple', 'banana', 'pear'.
        matrix = _padded_matrix(
            [
                ("fruit", [1.0, 0.0, 0.0]),
                ("apple", [1.0, 0.0, 0.0]),
                ("banana", [0.95, 0.05, 0.0]),
                ("pear", [0.9, 0.1, 0.0]),
                ("car", [-1.0, 0.0, 0.0]),
            ]
        )
        board = make_board(
            [
                ("apple", Color.RED, False),
                ("banana", Color.RED, False),
                ("pear", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=1.0)  # aggressive: never stop short
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("fruit", 3))

        assert trace.guesses[:3] == ("apple", "banana", "pear")

    def test_excludes_revealed_cards_from_candidates(self):
        matrix = _padded_matrix(
            [
                ("clue", [1.0, 0.0, 0.0]),
                ("apple", [1.0, 0.0, 0.0]),
                ("car", [-1.0, 0.0, 0.0]),
            ]
        )
        board = make_board(
            [
                ("apple", Color.RED, True),  # revealed → not a candidate
                ("car", Color.BLUE, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=1.0)
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 1))
        candidate_words = [c.word for c in trace.candidates]
        assert "apple" not in candidate_words

    def test_pass_clue_returns_no_guesses(self):
        matrix = _padded_matrix([("apple", [1.0, 0.0, 0.0])])
        board = make_board([("apple", Color.RED, False)])
        guesser = AIGuesser(matrix, risk=0.5)
        trace = guesser.guess(
            GuesserView(board=board, team=Color.RED), Clue(word="", count=0)
        )
        assert trace.guesses == ()
        assert trace.stop_reason == "pass_clue"

    def test_always_commits_to_pick_one(self):
        # Even when similarity is very low, pick #1 is always committed.
        matrix = _padded_matrix(
            [
                ("clue", [1.0, 0.0, 0.0]),
                ("weak", [-1.0, 0.0, 0.0]),  # actually negatively correlated
            ]
        )
        board = make_board([("weak", Color.RED, False)])
        guesser = AIGuesser(matrix, risk=0.0)  # very cautious
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 1))
        assert trace.guesses == ("weak",)


class TestStoppingPolicy:
    def test_stops_short_when_below_confidence_floor(self):
        # Clue 'fruit' targets 3 cards by spymaster's intent, but the 2nd ranked
        # card has a similarity well below the cautious floor.
        matrix = _padded_matrix(
            [
                ("fruit", [1.0, 0.0, 0.0]),
                ("apple", [1.0, 0.0, 0.0]),
                ("low", [0.05, 0.99, 0.0]),  # very low sim to 'fruit'
                ("car", [-1.0, 0.0, 0.0]),
            ]
        )
        board = make_board(
            [
                ("apple", Color.RED, False),
                ("low", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=0.0)  # cautious → high floor
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("fruit", 3))
        assert trace.guesses == ("apple",)
        assert trace.stop_reason == "confidence_floor"

    def test_aggressive_takes_all_n_even_with_low_sim(self):
        matrix = _padded_matrix(
            [
                ("fruit", [1.0, 0.0, 0.0]),
                ("apple", [1.0, 0.0, 0.0]),
                ("low", [0.05, 0.99, 0.0]),
            ]
        )
        board = make_board(
            [
                ("apple", Color.RED, False),
                ("low", Color.RED, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=1.0)
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("fruit", 2))
        assert trace.guesses == ("apple", "low")
        assert trace.stop_reason == "reached_n"


class TestBonusPick:
    def test_aggressive_takes_bonus_when_gap_is_small(self):
        # Three candidates almost equally close to clue. N=2, so a 3rd card
        # is comparable to the 2nd → bonus pick fires.
        matrix = _padded_matrix(
            [
                ("clue", [1.0, 0.0, 0.0]),
                ("a", [1.0, 0.01, 0.0]),
                ("b", [0.99, 0.02, 0.0]),
                ("c", [0.98, 0.03, 0.0]),
                ("far", [-1.0, 0.0, 0.0]),
            ]
        )
        board = make_board(
            [
                ("a", Color.RED, False),
                ("b", Color.RED, False),
                ("c", Color.RED, False),
                ("far", Color.BLUE, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=1.0)
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 2))
        assert len(trace.guesses) == 3
        assert trace.guesses[2] == "c"
        assert trace.bonus_attempted is True
        assert trace.stop_reason == "reached_n_plus_bonus"
        # The bonus pick is marked.
        committed_bonus = [c for c in trace.candidates if c.is_bonus]
        assert len(committed_bonus) == 1

    def test_cautious_never_takes_bonus(self):
        matrix = _padded_matrix(
            [
                ("clue", [1.0, 0.0, 0.0]),
                ("a", [1.0, 0.0, 0.0]),
                ("b", [0.99, 0.01, 0.0]),
                ("c", [0.98, 0.02, 0.0]),
            ]
        )
        board = make_board(
            [
                ("a", Color.RED, False),
                ("b", Color.RED, False),
                ("c", Color.RED, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=0.0)  # cautious, bonus disabled
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 2))
        assert len(trace.guesses) == 2
        assert trace.bonus_attempted is False

    def test_no_bonus_when_gap_too_large(self):
        # 3rd candidate is much further away → bonus skipped even on aggressive risk.
        matrix = _padded_matrix(
            [
                ("clue", [1.0, 0.0, 0.0]),
                ("a", [1.0, 0.0, 0.0]),
                ("b", [0.99, 0.01, 0.0]),
                ("c", [0.5, 0.5, 0.0]),  # big drop
            ]
        )
        board = make_board(
            [
                ("a", Color.RED, False),
                ("b", Color.RED, False),
                ("c", Color.RED, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=1.0)
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 2))
        assert len(trace.guesses) == 2
        assert trace.bonus_attempted is False


class TestValidation:
    def test_raises_on_unknown_clue_word(self):
        matrix = _padded_matrix([("apple", [1.0, 0.0, 0.0])])
        board = make_board([("apple", Color.RED, False)])
        guesser = AIGuesser(matrix, risk=0.5)
        with pytest.raises(ValueError, match="not in embedding matrix"):
            guesser.guess(
                GuesserView(board=board, team=Color.RED), Clue("missing", 1)
            )

    def test_raises_on_unknown_unrevealed_card(self):
        matrix = _padded_matrix([("clue", [1.0, 0.0, 0.0])])
        board = make_board([("ghost", Color.RED, False)])
        guesser = AIGuesser(matrix, risk=0.5)
        with pytest.raises(ValueError, match="missing from matrix"):
            guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 1))


class TestTraceShape:
    def test_candidates_sorted_descending_and_complete(self):
        matrix = _padded_matrix(
            [
                ("clue", [1.0, 0.0, 0.0]),
                ("a", [1.0, 0.0, 0.0]),
                ("b", [0.5, 0.5, 0.0]),
                ("c", [-1.0, 0.0, 0.0]),
            ]
        )
        board = make_board(
            [
                ("a", Color.RED, False),
                ("b", Color.RED, False),
                ("c", Color.BLUE, False),
            ]
        )
        guesser = AIGuesser(matrix, risk=0.5)
        trace = guesser.guess(GuesserView(board=board, team=Color.RED), Clue("clue", 1))
        # All 3 unrevealed cards present in candidates.
        words = [c.word for c in trace.candidates if c.word in ("a", "b", "c")]
        assert set(words) == {"a", "b", "c"}
        # Sorted by sim desc.
        sims = [c.similarity for c in trace.candidates]
        assert sims == sorted(sims, reverse=True)
