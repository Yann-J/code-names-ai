"""End-to-end single-turn test: spymaster gives a clue, guesser plays it.

Validates that the spymaster's intended targets are recoverable by an
embedding-only guesser using the same matrix — i.e. the two halves of the
pipeline agree about word similarity. This is the M4 deliverable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Board, Card, Clue, Color, GuesserView, SpymasterView
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / float(np.linalg.norm(v))


def test_spymaster_clue_drives_guesser_to_intended_targets():
    # Hand-crafted scenario: 9 friendly cards split into two clusters:
    # 5 "fruit-like" friends and 4 "vehicle-like" friends. Non-friendlies
    # all in unrelated directions. Two clue candidates: one fruit-aligned,
    # one vehicle-aligned. We expect the spymaster to pick whichever has
    # better margin and the guesser to recover the intended targets.
    entries: list[tuple[str, list[float], float]] = []
    for i in range(5):
        entries.append((f"fruit{i}", [1.0, 0.05 * i, 0.0], 5.0))
    for i in range(4):
        entries.append((f"car{i}", [0.0, 0.0, 1.0 + 0.05 * i], 5.0))
    for i in range(8):
        entries.append((f"opp{i}", [-1.0, -0.5 - 0.02 * i, 0.0], 5.0))
    for i in range(7):
        entries.append((f"neu{i}", [-1.0, 0.5 + 0.02 * i, 0.0], 5.0))
    entries.append(("ass", [-0.5, 0.0, -1.0], 5.0))
    entries.append(("PRODUCE", [1.0, 0.1, 0.0], 5.5))  # aligned with fruit cluster
    entries.append(("VEHICLE", [0.0, 0.0, 1.05], 5.5))  # aligned with car cluster

    surfaces = [s for s, _, _ in entries]
    vectors = np.stack([_normalize(v) for _, v, _ in entries])
    matrix = EmbeddingMatrix(
        vectors=vectors,
        surfaces=surfaces,
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id="test",
        vocab_cache_key="test",
    )
    df = pd.DataFrame(
        [{"surface": s, "lemma": s, "zipf": z, "pos": "NOUN"} for s, _, z in entries]
    )
    vocab = Vocabulary(
        config=VocabConfig(
            language="en",
            zipf_min=3.0,
            zipf_max=7.0,
            allowed_pos=frozenset({"NOUN"}),
        ),
        df=df,
    )

    friendly_words = [f"fruit{i}" for i in range(5)] + [f"car{i}" for i in range(4)]
    cards = (
        [Card(word=w, lemma=w, color=Color.RED) for w in friendly_words]
        + [Card(word=f"opp{i}", lemma=f"opp{i}", color=Color.BLUE) for i in range(8)]
        + [Card(word=f"neu{i}", lemma=f"neu{i}", color=Color.NEUTRAL) for i in range(7)]
        + [Card(word="ass", lemma="ass", color=Color.ASSASSIN)]
    )
    board = Board(cards=tuple(cards), first_team=Color.RED)

    spymaster = AISpymaster(matrix, vocab, risk=0.5)
    trace_sm = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))

    assert trace_sm.chosen is not None
    chosen = trace_sm.chosen
    # The spymaster should target one of the two clusters cleanly.
    assert chosen.clue in {"PRODUCE", "VEHICLE"}

    guesser = AIGuesser(matrix, risk=0.5)
    trace_g = guesser.guess(
        GuesserView(board=board, team=Color.RED),
        Clue(word=chosen.clue, count=chosen.n),
    )

    # The guesser's first N picks should match the spymaster's intended targets
    # (in some order — we don't pin the exact ordering).
    assert set(trace_g.guesses[: chosen.n]) == set(chosen.targets)


def test_pass_clue_yields_empty_guess_sequence():
    matrix = EmbeddingMatrix(
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        surfaces=["only"],
        surface_to_index={"only": 0},
        provider_id="test",
        vocab_cache_key="test",
    )
    board = Board(
        cards=tuple(
            [Card(word="only", lemma="only", color=Color.RED, revealed=True)]
            + [Card(word=f"x{i}", lemma=f"x{i}", color=Color.NEUTRAL, revealed=True) for i in range(24)]
        ),
        first_team=Color.RED,
    )
    guesser = AIGuesser(matrix, risk=0.5)
    trace = guesser.guess(
        GuesserView(board=board, team=Color.RED),
        Clue(word="", count=0),
    )
    assert trace.guesses == ()
    assert trace.stop_reason == "pass_clue"
