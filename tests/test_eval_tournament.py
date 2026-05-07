"""Tournament tests with real AI players on a synthetic embedding space."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.eval.tournament import run_tournament
from codenames_ai.game.models import Color
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / float(np.linalg.norm(v))


def _build():
    """Synthetic 3D embedding space + a 60-word vocab so boards can be sampled."""
    entries: list[tuple[str, list[float], float]] = []
    rng = np.random.default_rng(0)
    for i in range(30):
        v = rng.normal(size=3)
        entries.append((f"red{i}", list(v), 5.0))
    for i in range(30):
        v = rng.normal(size=3)
        entries.append((f"blue{i}", list(v), 5.0))
    # Plus a bunch of clue candidates spread across the unit sphere.
    for i in range(40):
        v = rng.normal(size=3)
        entries.append((f"clue{i}", list(v), 5.0))

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
    return matrix, vocab


def test_run_tournament_produces_one_record_per_seed():
    matrix, vocab = _build()
    spy = AISpymaster(matrix, vocab, risk=0.5)
    guesser = AIGuesser(matrix, risk=0.5)
    records = run_tournament(
        seeds=[1, 2, 3],
        game_vocab=vocab,
        red_spymaster=spy,
        red_guesser=guesser,
        blue_spymaster=spy,
        blue_guesser=guesser,
        max_clues=15,
    )
    assert len(records) == 3
    assert [r.seed for r in records] == [1, 2, 3]


def test_records_carry_label():
    matrix, vocab = _build()
    spy = AISpymaster(matrix, vocab, risk=0.5)
    guesser = AIGuesser(matrix, risk=0.5)
    records = run_tournament(
        seeds=[1],
        game_vocab=vocab,
        red_spymaster=spy,
        red_guesser=guesser,
        blue_spymaster=spy,
        blue_guesser=guesser,
        max_clues=15,
        label="risk=0.5",
    )
    assert records[0].label == "risk=0.5"


def test_record_properties():
    matrix, vocab = _build()
    spy = AISpymaster(matrix, vocab, risk=0.5)
    guesser = AIGuesser(matrix, risk=0.5)
    records = run_tournament(
        seeds=[42],
        game_vocab=vocab,
        red_spymaster=spy,
        red_guesser=guesser,
        blue_spymaster=spy,
        blue_guesser=guesser,
        max_clues=15,
    )
    r = records[0]
    assert r.num_clues > 0
    assert r.num_guesses > 0
    assert 0 <= r.correct_guesses <= r.num_guesses
    assert isinstance(r.assassin_hit, bool)
    assert r.first_team in (Color.RED, Color.BLUE)


def test_tournament_is_deterministic_for_same_seeds():
    matrix, vocab = _build()
    spy = AISpymaster(matrix, vocab, risk=0.5)
    guesser = AIGuesser(matrix, risk=0.5)
    common = dict(
        seeds=[1, 2, 3],
        game_vocab=vocab,
        red_spymaster=spy,
        red_guesser=guesser,
        blue_spymaster=spy,
        blue_guesser=guesser,
        max_clues=15,
    )
    a = run_tournament(**common)
    b = run_tournament(**common)
    for ra, rb in zip(a, b):
        assert ra.winner == rb.winner
        assert ra.num_clues == rb.num_clues
        assert ra.num_guesses == rb.num_guesses
