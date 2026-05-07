"""End-to-end smoke test against a real fastText model.

Skipped unless `CODENAMES_AI_FASTTEXT_PATH` points at an existing `.bin`. The
real English model is ~7GB; setting this env var is the dev opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codenames_ai.embedding import FastTextProvider, build_embedding_matrix
from codenames_ai.vocab.models import Vocabulary, VocabConfig


@pytest.fixture(scope="module")
def fasttext_path() -> Path:
    raw = os.environ.get("CODENAMES_AI_FASTTEXT_PATH")
    if not raw:
        pytest.skip("CODENAMES_AI_FASTTEXT_PATH not set")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"fastText model not at {path}")
    return path


def test_fasttext_projects_and_neighbors_are_sensible(fasttext_path):
    provider = FastTextProvider(fasttext_path)
    config = VocabConfig(
        language="en",
        zipf_min=4.0,
        zipf_max=7.0,
        allowed_pos=frozenset({"NOUN"}),
    )
    surfaces = ["apple", "banana", "pear", "car", "truck", "bicycle", "dog", "cat"]
    df = pd.DataFrame(
        [{"surface": s, "lemma": s, "zipf": 5.0, "pos": "NOUN"} for s in surfaces]
    )
    vocab = Vocabulary(config=config, df=df)

    matrix = build_embedding_matrix(vocab, provider)
    assert matrix.dim == provider.dim
    np.testing.assert_allclose(np.linalg.norm(matrix.vectors, axis=1), 1.0, atol=1e-5)

    # Sanity: 'apple' is closer to 'banana' than to 'truck'.
    assert matrix.sim("apple", "banana") > matrix.sim("apple", "truck")
    # 'car' is closer to 'truck' than to 'apple'.
    assert matrix.sim("car", "truck") > matrix.sim("car", "apple")
