"""End-to-end vocab build using real wordfreq + real spaCy.

Skipped when the spaCy English model isn't installed (`python -m spacy download
en_core_web_sm`). The wordfreq dep is always present per pyproject.
"""

from __future__ import annotations

import pytest

from codenames_ai.storage import StoragePaths
from codenames_ai.vocab import (
    SpacyLinguisticProcessor,
    VocabConfig,
    WordfreqProvider,
    load_or_build_vocabulary,
)


@pytest.fixture(scope="module")
def linguistic():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy 'en_core_web_sm' not installed")
    return SpacyLinguisticProcessor.for_language("en")


def test_wordfreq_provider_yields_descending_zipf():
    provider = WordfreqProvider()
    zipfs = [z for _, z in provider.iter_range(language="en", zipf_min=5.0, zipf_max=8.0)]
    assert zipfs, "expected at least some words in the [5.0, 8.0] window"
    assert zipfs == sorted(zipfs, reverse=True)


def test_build_clue_vocab_end_to_end(tmp_path, linguistic):
    storage = StoragePaths(cache_dir=tmp_path)
    config = VocabConfig(
        language="en",
        zipf_min=4.5,
        zipf_max=6.5,
        allowed_pos=frozenset({"NOUN", "ADJ"}),
    )
    vocab = load_or_build_vocabulary(
        config,
        storage,
        frequency_provider=WordfreqProvider(),
        linguistic=linguistic,
    )

    assert len(vocab) > 100, "expected a substantial vocabulary in this Zipf window"
    assert set(vocab.df["pos"].unique()) <= {"NOUN", "ADJ"}
    assert all(z >= 4.5 for z in vocab.df["zipf"])
    assert all(z <= 6.5 for z in vocab.df["zipf"])
    # Sanity: a few obvious words should appear.
    assert "apple" in set(vocab.surfaces)
