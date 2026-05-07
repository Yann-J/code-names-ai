"""Builder integration tests using fake providers — no spaCy/wordfreq required."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codenames_ai.storage import StoragePaths
from codenames_ai.vocab.builder import build_vocabulary, load_or_build_vocabulary
from codenames_ai.vocab.frequency import FrequencyProvider
from codenames_ai.vocab.linguistic import LinguisticProcessor
from codenames_ai.vocab.models import VocabConfig


class FakeFrequencyProvider(FrequencyProvider):
    """In-memory list of `(word, zipf)` pairs, returned in the order given."""

    def __init__(self, entries: list[tuple[str, float]]) -> None:
        self.entries = entries
        self.calls = 0

    def iter_range(
        self, *, language: str, zipf_min: float, zipf_max: float
    ) -> Iterator[tuple[str, float]]:
        self.calls += 1
        for surface, zipf in self.entries:
            if zipf < zipf_min:
                return
            if zipf > zipf_max:
                continue
            yield surface, zipf


class FakeLinguistic(LinguisticProcessor):
    """Lookup-table-based POS+lemma — keeps tests deterministic without spaCy."""

    def __init__(self, table: dict[str, tuple[str, str]]) -> None:
        self.table = table

    def analyze_batch(self, words: list[str]) -> list[tuple[str, str]]:
        return [self.table.get(w, (w, "X")) for w in words]


def _config(tmp_path: Path | None = None, **overrides) -> VocabConfig:
    base = dict(
        language="en",
        zipf_min=3.0,
        zipf_max=7.0,
        allowed_pos=frozenset({"NOUN", "ADJ"}),
        min_length=3,
        allow_hyphens=True,
        exclusions_path=None,
    )
    base.update(overrides)
    return VocabConfig(**base)


class TestBuildVocabulary:
    def test_filters_by_zipf_range(self):
        provider = FakeFrequencyProvider(
            [("the", 7.5), ("apple", 5.0), ("rare", 3.5), ("obscure", 2.0)]
        )
        ling = FakeLinguistic(
            {
                "the": ("the", "DET"),
                "apple": ("apple", "NOUN"),
                "rare": ("rare", "ADJ"),
                "obscure": ("obscure", "ADJ"),
            }
        )
        vocab = build_vocabulary(
            _config(zipf_min=3.0, zipf_max=7.0),
            frequency_provider=provider,
            linguistic=ling,
        )
        assert set(vocab.surfaces) == {"apple", "rare"}

    def test_filters_by_pos(self):
        provider = FakeFrequencyProvider([("apple", 5.0), ("the", 6.0), ("frozen", 4.5)])
        ling = FakeLinguistic(
            {
                "apple": ("apple", "NOUN"),
                "the": ("the", "DET"),
                "frozen": ("freeze", "ADJ"),
            }
        )
        vocab = build_vocabulary(
            _config(allowed_pos=frozenset({"NOUN"})),
            frequency_provider=provider,
            linguistic=ling,
        )
        assert set(vocab.surfaces) == {"apple"}

    def test_filters_by_exclusions(self, tmp_path):
        exclusions = tmp_path / "ex.txt"
        exclusions.write_text("apple\n")
        provider = FakeFrequencyProvider([("apple", 5.0), ("banana", 4.5)])
        ling = FakeLinguistic({"apple": ("apple", "NOUN"), "banana": ("banana", "NOUN")})
        vocab = build_vocabulary(
            _config(exclusions_path=exclusions),
            frequency_provider=provider,
            linguistic=ling,
        )
        assert set(vocab.surfaces) == {"banana"}

    def test_filters_by_surface_validity(self):
        # apostrophe rejected, length<3 rejected, valid words pass
        provider = FakeFrequencyProvider(
            [("don't", 6.0), ("be", 6.0), ("apple", 5.0), ("well-being", 4.0)]
        )
        ling = FakeLinguistic(
            {
                "don't": ("do", "VERB"),
                "be": ("be", "VERB"),
                "apple": ("apple", "NOUN"),
                "well-being": ("well-being", "NOUN"),
            }
        )
        vocab = build_vocabulary(
            _config(),
            frequency_provider=provider,
            linguistic=ling,
        )
        assert set(vocab.surfaces) == {"apple", "well-being"}

    def test_dataframe_has_expected_columns_and_values(self):
        provider = FakeFrequencyProvider([("apple", 5.0), ("frozen", 4.5)])
        ling = FakeLinguistic(
            {"apple": ("apple", "NOUN"), "frozen": ("freeze", "ADJ")}
        )
        vocab = build_vocabulary(_config(), frequency_provider=provider, linguistic=ling)
        assert list(vocab.df.columns) == ["surface", "lemma", "zipf", "pos"]
        rows = vocab.df.set_index("surface").to_dict("index")
        assert rows["apple"] == {"lemma": "apple", "zipf": 5.0, "pos": "NOUN"}
        assert rows["frozen"] == {"lemma": "freeze", "zipf": 4.5, "pos": "ADJ"}

    def test_lowercases_surface_from_provider(self):
        provider = FakeFrequencyProvider([("Apple", 5.0)])
        ling = FakeLinguistic({"apple": ("apple", "NOUN")})
        vocab = build_vocabulary(_config(), frequency_provider=provider, linguistic=ling)
        assert vocab.surfaces == ["apple"]

    def test_empty_input_yields_empty_vocab(self):
        provider = FakeFrequencyProvider([])
        ling = FakeLinguistic({})
        vocab = build_vocabulary(_config(), frequency_provider=provider, linguistic=ling)
        assert len(vocab) == 0
        assert list(vocab.df.columns) == ["surface", "lemma", "zipf", "pos"]


class TestLoadOrBuild:
    def test_first_call_builds_and_writes_artifact(self, tmp_path):
        provider = FakeFrequencyProvider([("apple", 5.0)])
        ling = FakeLinguistic({"apple": ("apple", "NOUN")})
        config = _config()
        storage = StoragePaths(cache_dir=tmp_path)

        vocab = load_or_build_vocabulary(
            config, storage, frequency_provider=provider, linguistic=ling
        )
        assert vocab.surfaces == ["apple"]
        assert provider.calls == 1
        cache_path = storage.vocab_dir_for("en") / f"{config.cache_key()}.parquet"
        assert cache_path.exists()

    def test_second_call_loads_from_cache_without_invoking_providers(self, tmp_path):
        provider = FakeFrequencyProvider([("apple", 5.0)])
        ling = FakeLinguistic({"apple": ("apple", "NOUN")})
        config = _config()
        storage = StoragePaths(cache_dir=tmp_path)

        load_or_build_vocabulary(
            config, storage, frequency_provider=provider, linguistic=ling
        )
        assert provider.calls == 1

        load_or_build_vocabulary(
            config, storage, frequency_provider=provider, linguistic=ling
        )
        # Cache hit: provider not re-invoked.
        assert provider.calls == 1

    def test_different_configs_produce_different_artifacts(self, tmp_path):
        provider = FakeFrequencyProvider([("apple", 5.0), ("frozen", 4.5)])
        ling = FakeLinguistic(
            {"apple": ("apple", "NOUN"), "frozen": ("freeze", "ADJ")}
        )
        storage = StoragePaths(cache_dir=tmp_path)

        nouns_only = _config(allowed_pos=frozenset({"NOUN"}))
        nouns_and_adj = _config(allowed_pos=frozenset({"NOUN", "ADJ"}))

        a = load_or_build_vocabulary(
            nouns_only, storage, frequency_provider=provider, linguistic=ling
        )
        b = load_or_build_vocabulary(
            nouns_and_adj, storage, frequency_provider=provider, linguistic=ling
        )
        assert set(a.surfaces) == {"apple"}
        assert set(b.surfaces) == {"apple", "frozen"}

        assert (storage.vocab_dir_for("en") / f"{nouns_only.cache_key()}.parquet").exists()
        assert (storage.vocab_dir_for("en") / f"{nouns_and_adj.cache_key()}.parquet").exists()
