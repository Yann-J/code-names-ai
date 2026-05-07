from collections import Counter

import pandas as pd
import pytest

from codenames_ai.game.board import generate_board
from codenames_ai.game.models import Color
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _vocab(n: int = 50) -> Vocabulary:
    config = VocabConfig(
        language="en",
        zipf_min=4.0,
        zipf_max=6.5,
        allowed_pos=frozenset({"NOUN"}),
    )
    df = pd.DataFrame(
        [
            {
                "surface": f"word{i:03d}",
                "lemma": f"word{i:03d}",
                "zipf": 5.0,
                "pos": "NOUN",
            }
            for i in range(n)
        ]
    )
    return Vocabulary(config=config, df=df)


class TestGenerateBoard:
    def test_returns_25_cards(self):
        board = generate_board(_vocab(), seed=1)
        assert len(board.cards) == 25

    def test_color_counts(self):
        board = generate_board(_vocab(), seed=1)
        counts = Counter(c.color for c in board.cards)
        # 9 first_team, 8 other_team, 7 neutral, 1 assassin → check via shape
        team_counts = sorted([counts[Color.RED], counts[Color.BLUE]])
        assert team_counts == [8, 9]
        assert counts[Color.NEUTRAL] == 7
        assert counts[Color.ASSASSIN] == 1

    def test_first_team_has_nine_cards(self):
        board = generate_board(_vocab(), seed=1, first_team=Color.RED)
        red_count = sum(1 for c in board.cards if c.color == Color.RED)
        blue_count = sum(1 for c in board.cards if c.color == Color.BLUE)
        assert red_count == 9
        assert blue_count == 8
        assert board.first_team == Color.RED

    def test_first_team_blue_has_nine_cards(self):
        board = generate_board(_vocab(), seed=1, first_team=Color.BLUE)
        red_count = sum(1 for c in board.cards if c.color == Color.RED)
        blue_count = sum(1 for c in board.cards if c.color == Color.BLUE)
        assert blue_count == 9
        assert red_count == 8

    def test_deterministic_for_same_seed(self):
        a = generate_board(_vocab(), seed=42)
        b = generate_board(_vocab(), seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_board(_vocab(), seed=1)
        b = generate_board(_vocab(), seed=2)
        assert a != b

    def test_first_team_chosen_from_seed_when_unspecified(self):
        # Same seed → same auto-picked first_team.
        a = generate_board(_vocab(), seed=99)
        b = generate_board(_vocab(), seed=99)
        assert a.first_team == b.first_team

    def test_no_duplicate_words(self):
        board = generate_board(_vocab(), seed=7)
        words = [c.word for c in board.cards]
        assert len(words) == len(set(words))

    def test_lemma_carried_from_vocab(self):
        config = VocabConfig(
            language="en",
            zipf_min=4.0,
            zipf_max=6.5,
            allowed_pos=frozenset({"NOUN"}),
        )
        df = pd.DataFrame(
            [
                {"surface": f"surf{i}", "lemma": f"lemma{i}", "zipf": 5.0, "pos": "NOUN"}
                for i in range(30)
            ]
        )
        vocab = Vocabulary(config=config, df=df)
        board = generate_board(vocab, seed=3)
        for card in board.cards:
            n = card.word.removeprefix("surf")
            assert card.lemma == f"lemma{n}"

    def test_raises_when_vocab_too_small(self):
        small_vocab = _vocab(n=20)
        with pytest.raises(ValueError, match="at least 25"):
            generate_board(small_vocab, seed=1)

    def test_raises_when_first_team_invalid(self):
        with pytest.raises(ValueError, match="RED or BLUE"):
            generate_board(_vocab(), seed=1, first_team=Color.NEUTRAL)
