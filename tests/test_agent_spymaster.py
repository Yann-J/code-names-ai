"""End-to-end tests for AISpymaster using hand-crafted synthetic embeddings.

Each test constructs a small `EmbeddingMatrix` + `Vocabulary` where the right
answer is computable by inspection, then asserts the spymaster picks it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from codenames_ai.agent.interfaces import NoLegalClueError
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.spymaster import AISpymaster, _ClueIndex
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Board, Card, Color, SpymasterView
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v


def make_setup(
    entries: list[tuple[str, list[float], float]],
) -> tuple[EmbeddingMatrix, Vocabulary]:
    """Build a matched (matrix, vocab) pair from `(surface, vector, zipf)` triples."""
    surfaces = [e[0] for e in entries]
    vectors = np.stack([_normalize(e[1]) for e in entries])
    matrix = EmbeddingMatrix(
        vectors=vectors,
        surfaces=surfaces,
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id="test",
        vocab_cache_key="test",
    )
    config = VocabConfig(
        language="en",
        zipf_min=3.0,
        zipf_max=7.0,
        allowed_pos=frozenset({"NOUN"}),
    )
    df = pd.DataFrame(
        [{"surface": s, "lemma": s, "zipf": z, "pos": "NOUN"} for (s, _, z) in entries]
    )
    return matrix, Vocabulary(config=config, df=df)


def make_board(
    *,
    friendly: list[str],
    opponent: list[str],
    neutral: list[str],
    assassin: str,
    first_team: Color = Color.RED,
) -> Board:
    assert len(friendly) == 9
    assert len(opponent) == 8
    assert len(neutral) == 7
    other = first_team.opponent()
    cards = (
        [Card(word=w, lemma=w, color=first_team) for w in friendly]
        + [Card(word=w, lemma=w, color=other) for w in opponent]
        + [Card(word=w, lemma=w, color=Color.NEUTRAL) for w in neutral]
        + [Card(word=assassin, lemma=assassin, color=Color.ASSASSIN)]
    )
    return Board(cards=tuple(cards), first_team=first_team)


def _board_words(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _basic_setup(extra_clues: list[tuple[str, list[float], float]] | None = None):
    """A board where friendlies cluster in one direction, non-friendlies the other."""
    friendly_dir = [1.0, 0.0]
    nonfriendly_dir = [-1.0, 0.0]

    entries: list[tuple[str, list[float], float]] = []
    for i, w in enumerate(_board_words("f", 9)):
        # Slight variation so they aren't degenerate.
        entries.append((w, [1.0, 0.01 * i], 5.0))
    for i, w in enumerate(_board_words("o", 8)):
        entries.append((w, [-1.0, 0.01 * i], 5.0))
    for i, w in enumerate(_board_words("n", 7)):
        entries.append((w, [-1.0, 0.5 + 0.01 * i], 5.0))
    entries.append(("ass", [-1.0, -0.3], 5.0))

    extra_clues = extra_clues or []
    entries.extend(extra_clues)
    return entries, friendly_dir, nonfriendly_dir


class TestBasicBehavior:
    def test_picks_clue_pointing_at_friendly_cluster(self):
        entries, _, _ = _basic_setup(
            extra_clues=[
                ("good_clue", [1.0, 0.0], 5.0),  # aligned with friendlies
                ("bad_clue", [-1.0, 0.0], 5.0),  # aligned with non-friendlies
            ]
        )
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(
            matrix,
            vocab,
            risk=0.5,
            weights=replace(ScoringWeights.from_risk(0.5), lane_max_n=9),
        )
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))

        assert trace.chosen is not None
        assert trace.chosen.clue == "good_clue"
        assert trace.chosen.n == 9

    def test_top_candidates_sorted_descending(self):
        entries, _, _ = _basic_setup(
            extra_clues=[
                ("clue_a", [1.0, 0.0], 5.0),
                ("clue_b", [0.95, 0.05], 5.0),
                ("clue_c", [0.5, 0.5], 5.0),
            ]
        )
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))

        scores = [c.score for c in trace.top_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_top_candidates_unique_clue_surfaces(self):
        entries, _, _ = _basic_setup(extra_clues=[("good_clue", [1.0, 0.0], 5.0)])
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        clues = [c.clue for c in trace.top_candidates]
        assert len(clues) == len(set(clues))

    def test_targets_match_top_friendlies(self):
        entries, _, _ = _basic_setup(extra_clues=[("clue", [1.0, 0.0], 5.0)])
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))

        # All friendly cards are roughly aligned; the chosen subset is a prefix
        # of the sorted-by-similarity list (whatever order spaCy/numpy produced).
        assert set(trace.chosen.targets).issubset(set(_board_words("f", 9)))
        assert len(trace.chosen.targets) == trace.chosen.n


class TestLegality:
    def test_filters_clue_matching_board_word(self):
        entries, _, _ = _basic_setup(
            extra_clues=[
                (
                    "f0",
                    [1.0, 0.0],
                    5.0,
                ),  # would be a great clue but matches a board word
                ("good_clue", [1.0, 0.0], 5.0),
            ]
        )
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        assert trace.chosen.clue == "good_clue"
        assert trace.illegal_count > 0

    def test_filters_clue_matching_via_substring(self):
        # 'flame' contains 'lame' (board word) — should be vetoed under
        # default lemma_substring strictness.
        entries, _, _ = _basic_setup(
            extra_clues=[("flame", [1.0, 0.0], 5.0), ("good", [1.0, 0.0], 5.0)]
        )
        # Replace one friendly with 'lame' so the substring check fires.
        # We need to rebuild from scratch to splice that in.
        replaced = []
        for s, v, z in entries:
            if s == "f0":
                replaced.append(("lame", v, z))
            else:
                replaced.append((s, v, z))
        matrix, vocab = make_setup(replaced)
        friendly = ["lame"] + _board_words("f", 9)[1:]
        board = make_board(
            friendly=friendly,
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        assert trace.chosen.clue == "good"


class TestVetoes:
    def test_no_legal_clue_raises(self):
        # Every candidate clue is right next to the assassin → ceiling vetoes.
        entries, _, _ = _basic_setup(extra_clues=[("near_assassin", [-1.0, -0.3], 5.0)])
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.0)  # cautious
        with pytest.raises(NoLegalClueError):
            spymaster.give_clue(SpymasterView(board=board, team=Color.RED))

    def test_validates_board_words_are_in_matrix(self):
        # vocab/matrix doesn't include 'mystery' but the board does.
        entries = [(f"f{i}", [1.0, 0.01 * i], 5.0) for i in range(9)]
        entries += [(f"o{i}", [-1.0, 0.01 * i], 5.0) for i in range(8)]
        entries += [(f"n{i}", [-1.0, 0.5 + 0.01 * i], 5.0) for i in range(7)]
        entries.append(("ass", [-1.0, -0.3], 5.0))
        entries.append(("clue", [1.0, 0.0], 5.0))
        matrix, vocab = make_setup(entries)

        # Splice in a board word missing from the matrix.
        cards = list(
            make_board(
                friendly=_board_words("f", 9),
                opponent=_board_words("o", 8),
                neutral=_board_words("n", 7),
                assassin="ass",
            ).cards
        )
        cards[0] = Card(word="mystery", lemma="mystery", color=Color.RED)
        board = Board(cards=tuple(cards), first_team=Color.RED)

        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        with pytest.raises(ValueError, match="missing from embedding matrix"):
            spymaster.give_clue(SpymasterView(board=board, team=Color.RED))


class TestRiskKnob:
    def test_aggressive_prefers_higher_n_than_cautious(self):
        # Setup: one clue covers 3 friendlies tightly, another covers 1 with much bigger margin.
        # Friendlies split into 'tight' cluster and 'isolated' card.
        entries: list[tuple[str, list[float], float]] = []
        # 3 tight friendlies clustered around (1, 0)
        entries.append(("ft0", [1.0, 0.0], 5.0))
        entries.append(("ft1", [1.0, 0.05], 5.0))
        entries.append(("ft2", [1.0, -0.05], 5.0))
        # 1 isolated friendly at (0, 1) — completely separate direction
        entries.append(("fi0", [0.0, 1.0], 5.0))
        # 5 friendlies scattered (so we still have 9), at varying positions
        for i in range(5):
            entries.append((f"fs{i}", [-0.2, 0.5 + 0.05 * i], 5.0))
        # Non-friendlies far away in (-1, 0) hemisphere
        for i in range(8):
            entries.append((f"o{i}", [-1.0, -0.1 * i / 7], 5.0))
        for i in range(7):
            entries.append((f"n{i}", [-0.8, -0.5 - 0.05 * i], 5.0))
        entries.append(("ass", [-1.0, 0.5], 5.0))
        # Two clue candidates:
        # 'ambitious' targets the tight cluster (3 cards) at thin margin
        entries.append(("ambitious", [1.0, 0.0], 5.0))
        # 'safe' targets the isolated friendly with huge margin
        entries.append(("safe", [0.0, 1.0], 5.0))

        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=["ft0", "ft1", "ft2", "fi0"] + [f"fs{i}" for i in range(5)],
            opponent=[f"o{i}" for i in range(8)],
            neutral=[f"n{i}" for i in range(7)],
            assassin="ass",
        )

        cautious = AISpymaster(matrix, vocab, risk=0.0)
        aggressive = AISpymaster(matrix, vocab, risk=1.0)

        cautious_trace = cautious.give_clue(SpymasterView(board=board, team=Color.RED))
        aggressive_trace = aggressive.give_clue(
            SpymasterView(board=board, team=Color.RED)
        )

        # Aggressive should pick higher-N (or at least not lower) than cautious.
        assert aggressive_trace.chosen.n >= cautious_trace.chosen.n


class TestExplicitWeightOverride:
    def test_explicit_weights_take_precedence_over_risk(self):
        weights = ScoringWeights.from_risk(0.5)
        entries, _, _ = _basic_setup(extra_clues=[("clue", [1.0, 0.0], 5.0)])
        matrix, vocab = make_setup(entries)
        spymaster = AISpymaster(matrix, vocab, risk=0.0, weights=weights)
        assert spymaster.weights == weights


class TestClueIndexDedupe:
    def test_collapses_duplicate_vocab_surfaces(self):
        """Vocab rows can repeat the same clue surface (e.g. POS paths); index once."""
        z = 5.0
        entries: list[tuple[str, list[float], float]] = []
        for i in range(9):
            entries.append((f"f{i}", [1.0, 0.01 * i], z))
        for i in range(8):
            entries.append((f"o{i}", [-1.0, 0.01 * i], z))
        for i in range(7):
            entries.append((f"n{i}", [-1.0, 0.5 + 0.01 * i], z))
        entries.append(("ass", [-1.0, -0.3], z))
        entries.append(("dupclue", [1.0, 0.0], z))
        matrix, _ = make_setup(entries)
        cfg = VocabConfig(
            language="en",
            zipf_min=3.0,
            zipf_max=7.0,
            allowed_pos=frozenset({"NOUN"}),
        )
        df = pd.DataFrame(
            [
                {"surface": "dupclue", "lemma": "a", "zipf": 4.0, "pos": "NOUN"},
                {"surface": "dupclue", "lemma": "b", "zipf": 6.5, "pos": "NOUN"},
                {"surface": "otherclue", "lemma": "c", "zipf": 5.0, "pos": "NOUN"},
            ]
        )
        vocab = Vocabulary(config=cfg, df=df)
        idx = _ClueIndex.build(vocab, matrix)
        assert idx.surfaces.count("dupclue") == 1
        i = idx.surfaces.index("dupclue")
        assert idx.zipfs[i] == pytest.approx(6.5)
        assert idx.lemmas[i] == "b"


class TestComponentsBalance:
    def test_score_equals_components_total(self):
        entries, _, _ = _basic_setup(extra_clues=[("clue", [1.0, 0.0], 5.0)])
        matrix, vocab = make_setup(entries)
        board = make_board(
            friendly=_board_words("f", 9),
            opponent=_board_words("o", 8),
            neutral=_board_words("n", 7),
            assassin="ass",
        )
        spymaster = AISpymaster(matrix, vocab, risk=0.5)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        for cand in trace.top_candidates:
            assert cand.score == pytest.approx(cand.components.total)
