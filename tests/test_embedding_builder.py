import numpy as np
import pandas as pd
import pytest

from codenames_ai.embedding.builder import (
    build_embedding_matrix,
    load_or_build_embedding_matrix,
)
from codenames_ai.embedding.provider import EmbeddingProvider
from codenames_ai.storage import StoragePaths
from codenames_ai.vocab.models import Vocabulary, VocabConfig


class FakeProvider(EmbeddingProvider):
    """Returns deterministic vectors keyed by surface form."""

    def __init__(
        self,
        table: dict[str, list[float]],
        *,
        provider_id: str = "fake",
        dim: int = 3,
    ) -> None:
        self.table = table
        self._provider_id = provider_id
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def vectorize(self, surfaces):
        self.calls += 1
        return np.array([self.table[s] for s in surfaces], dtype=np.float32)


def _vocab(surfaces, *, language="en"):
    config = VocabConfig(
        language=language,
        zipf_min=3.0,
        zipf_max=7.0,
        allowed_pos=frozenset({"NOUN"}),
    )
    df = pd.DataFrame(
        [{"surface": s, "lemma": s, "zipf": 5.0, "pos": "NOUN"} for s in surfaces],
        columns=["surface", "lemma", "zipf", "pos"],
    )
    return Vocabulary(config=config, df=df)


class TestBuildEmbeddingMatrix:
    def test_projects_every_surface(self):
        provider = FakeProvider(
            {"apple": [1.0, 0.0, 0.0], "banana": [0.0, 1.0, 0.0]}
        )
        vocab = _vocab(["apple", "banana"])
        matrix = build_embedding_matrix(vocab, provider)
        assert matrix.surfaces == ["apple", "banana"]
        assert matrix.dim == 3

    def test_normalizes_vectors(self):
        provider = FakeProvider({"a": [3.0, 4.0, 0.0]})
        vocab = _vocab(["a"])
        matrix = build_embedding_matrix(vocab, provider)
        # original norm was 5; after normalization vector should have norm 1.
        np.testing.assert_allclose(np.linalg.norm(matrix["a"]), 1.0, atol=1e-6)
        np.testing.assert_allclose(matrix["a"], [0.6, 0.8, 0.0], atol=1e-6)

    def test_handles_zero_vector_without_nan(self):
        provider = FakeProvider({"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]})
        vocab = _vocab(["a", "b"])
        matrix = build_embedding_matrix(vocab, provider)
        assert not np.isnan(matrix["a"]).any()

    def test_carries_provider_and_vocab_keys(self):
        provider = FakeProvider({"a": [1.0, 0.0, 0.0]}, provider_id="prov-99")
        vocab = _vocab(["a"])
        matrix = build_embedding_matrix(vocab, provider)
        assert matrix.provider_id == "prov-99"
        assert matrix.vocab_cache_key == vocab.config.cache_key()

    def test_empty_vocab_yields_empty_matrix(self):
        provider = FakeProvider({}, dim=3)
        vocab = _vocab([])
        matrix = build_embedding_matrix(vocab, provider)
        assert len(matrix) == 0
        assert matrix.dim == 3


class TestLoadOrBuild:
    def test_first_call_builds_and_writes(self, tmp_path):
        provider = FakeProvider({"a": [1.0, 0.0, 0.0]})
        vocab = _vocab(["a"])
        storage = StoragePaths(cache_dir=tmp_path)

        matrix = load_or_build_embedding_matrix(vocab, provider, storage)
        assert matrix.surfaces == ["a"]
        assert provider.calls == 1
        # File exists in language-scoped subdir.
        files = list(storage.embed_dir_for("en").glob("*.npz"))
        assert len(files) == 1

    def test_second_call_loads_from_cache(self, tmp_path):
        provider = FakeProvider({"a": [1.0, 0.0, 0.0]})
        vocab = _vocab(["a"])
        storage = StoragePaths(cache_dir=tmp_path)

        load_or_build_embedding_matrix(vocab, provider, storage)
        load_or_build_embedding_matrix(vocab, provider, storage)
        # Provider only called once — second call hit the cache.
        assert provider.calls == 1

    def test_different_providers_produce_different_artifacts(self, tmp_path):
        vocab = _vocab(["a"])
        storage = StoragePaths(cache_dir=tmp_path)

        prov_a = FakeProvider({"a": [1.0, 0.0, 0.0]}, provider_id="prov-a")
        prov_b = FakeProvider({"a": [0.0, 1.0, 0.0]}, provider_id="prov-b")
        load_or_build_embedding_matrix(vocab, prov_a, storage)
        load_or_build_embedding_matrix(vocab, prov_b, storage)

        files = list(storage.embed_dir_for("en").glob("*.npz"))
        assert len(files) == 2

    def test_different_vocab_configs_produce_different_artifacts(self, tmp_path):
        # Cache key is content-addressed by VocabConfig — change the config
        # (here, language) and the artifact lives in a different file.
        provider = FakeProvider(
            {"a": [1.0, 0.0, 0.0]},
            provider_id="prov",
        )
        storage = StoragePaths(cache_dir=tmp_path)

        load_or_build_embedding_matrix(_vocab(["a"], language="en"), provider, storage)
        load_or_build_embedding_matrix(_vocab(["a"], language="fr"), provider, storage)

        en_files = list(storage.embed_dir_for("en").glob("*.npz"))
        fr_files = list(storage.embed_dir_for("fr").glob("*.npz"))
        assert len(en_files) == 1
        assert len(fr_files) == 1
