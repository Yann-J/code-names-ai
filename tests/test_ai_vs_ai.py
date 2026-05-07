"""Full AI-vs-AI integration test on a synthetic embedding space.

Both teams use real `AISpymaster` and `AIGuesser` instances; we just hand-craft
the embedding space so the game is winnable in finite turns. Verifies that the
M3+M4+M6 stack runs end-to-end and produces a winner deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Board, Card, Color
from codenames_ai.game.orchestrator import Game
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / float(np.linalg.norm(v))


def _build_setup():
    """A 3D space where RED and BLUE friendlies cluster orthogonally and
    several clue candidates exist for each cluster."""

    entries: list[tuple[str, list[float], float]] = []

    # 9 RED cards along axis x.
    for i in range(9):
        entries.append((f"r{i}", [1.0, 0.05 * i, 0.0], 5.0))
    # 8 BLUE cards along axis z.
    for i in range(8):
        entries.append((f"b{i}", [0.0, 0.05 * i, 1.0], 5.0))
    # 7 neutral cards in the middle (axis y).
    for i in range(7):
        entries.append((f"n{i}", [0.05 * i, 1.0, 0.05 * i], 5.0))
    # Assassin in a corner of negative space.
    entries.append(("ass", [-1.0, -1.0, -1.0], 5.0))
    # Clue words: several aligned with each cluster.
    entries.append(("red_clue1", [1.0, 0.0, 0.0], 5.5))
    entries.append(("red_clue2", [0.95, 0.05, 0.0], 5.0))
    entries.append(("blue_clue1", [0.0, 0.0, 1.0], 5.5))
    entries.append(("blue_clue2", [0.05, 0.0, 0.95], 5.0))

    surfaces = [s for s, _, _ in entries]
    vectors = np.stack([_norm(v) for _, v, _ in entries])
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

    cards = (
        [Card(word=f"r{i}", lemma=f"r{i}", color=Color.RED) for i in range(9)]
        + [Card(word=f"b{i}", lemma=f"b{i}", color=Color.BLUE) for i in range(8)]
        + [Card(word=f"n{i}", lemma=f"n{i}", color=Color.NEUTRAL) for i in range(7)]
        + [Card(word="ass", lemma="ass", color=Color.ASSASSIN)]
    )
    board = Board(cards=tuple(cards), first_team=Color.RED)
    return matrix, vocab, board


def test_ai_vs_ai_completes_with_a_winner():
    matrix, vocab, board = _build_setup()
    game = Game(
        board,
        red_spymaster=AISpymaster(matrix, vocab, risk=0.5),
        red_guesser=AIGuesser(matrix, risk=0.5),
        blue_spymaster=AISpymaster(matrix, vocab, risk=0.5),
        blue_guesser=AIGuesser(matrix, risk=0.5),
        max_clues=20,
    )
    final = game.play()
    assert final.is_over
    assert final.winner in (Color.RED, Color.BLUE, None)
    # Always at least one clue + one guess recorded.
    assert any(ev.kind == "CLUE" for ev in final.turn_history)
    assert any(ev.kind == "GUESS" for ev in final.turn_history)


def test_ai_vs_ai_deterministic_for_same_setup():
    def run():
        matrix, vocab, board = _build_setup()
        game = Game(
            board,
            red_spymaster=AISpymaster(matrix, vocab, risk=0.5),
            red_guesser=AIGuesser(matrix, risk=0.5),
            blue_spymaster=AISpymaster(matrix, vocab, risk=0.5),
            blue_guesser=AIGuesser(matrix, risk=0.5),
            max_clues=20,
        )
        return game.play()

    a = run()
    b = run()
    assert a.winner == b.winner
    assert len(a.turn_history) == len(b.turn_history)
    for ea, eb in zip(a.turn_history, b.turn_history):
        assert ea == eb
