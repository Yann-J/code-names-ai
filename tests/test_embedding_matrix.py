import json

import numpy as np
import pytest

from codenames_ai.embedding.matrix import EmbeddingMatrix


def _matrix(surfaces, vectors, *, provider_id="test", vocab_cache_key="abc"):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype(np.float32)
    return EmbeddingMatrix(
        vectors=vectors,
        surfaces=list(surfaces),
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id=provider_id,
        vocab_cache_key=vocab_cache_key,
    )


class TestConstruction:
    def test_rejects_non_float32(self):
        with pytest.raises(ValueError, match="float32"):
            EmbeddingMatrix(
                vectors=np.array([[1.0, 0.0]], dtype=np.float64),
                surfaces=["a"],
                surface_to_index={"a": 0},
                provider_id="t",
                vocab_cache_key="k",
            )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="rows"):
            EmbeddingMatrix(
                vectors=np.zeros((2, 3), dtype=np.float32),
                surfaces=["a"],
                surface_to_index={"a": 0},
                provider_id="t",
                vocab_cache_key="k",
            )

    def test_dim_and_len(self):
        m = _matrix(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        assert m.dim == 3
        assert len(m) == 2

    def test_contains_and_getitem(self):
        m = _matrix(["a", "b"], [[1, 0], [0, 1]])
        assert "a" in m
        assert "z" not in m
        np.testing.assert_allclose(m["a"], [1.0, 0.0])
        with pytest.raises(KeyError):
            _ = m["missing"]


class TestSimilarity:
    def test_sim_of_orthogonal_is_zero(self):
        m = _matrix(["a", "b"], [[1, 0], [0, 1]])
        assert m.sim("a", "b") == pytest.approx(0.0)

    def test_sim_of_identical_is_one(self):
        m = _matrix(["a", "b"], [[1, 0], [1, 0]])
        assert m.sim("a", "b") == pytest.approx(1.0)

    def test_sim_matrix_self(self):
        m = _matrix(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]])
        sm = m.sim_matrix(["a", "b", "c"])
        # diagonal == 1
        np.testing.assert_allclose(np.diag(sm), [1.0, 1.0, 1.0])
        # symmetry
        np.testing.assert_allclose(sm, sm.T)

    def test_sim_matrix_two_axes(self):
        m = _matrix(["a", "b", "c"], [[1, 0], [0, 1], [1, 0]])
        sm = m.sim_matrix(["a", "b"], ["a", "c"])
        # a-a == 1, a-c == 1 (parallel), b-a == 0, b-c == 0
        np.testing.assert_allclose(sm, [[1.0, 1.0], [0.0, 0.0]], atol=1e-6)

    def test_sim_matrix_raises_on_unknown_surface(self):
        m = _matrix(["a"], [[1, 0]])
        with pytest.raises(KeyError):
            m.sim_matrix(["a"], ["missing"])


class TestNearest:
    def test_returns_top_k_sorted_desc(self):
        # 'apple' is closest to 'fruit' (parallel), then 'pear', then 'car'.
        m = _matrix(
            ["apple", "fruit", "pear", "car"],
            [[1, 0], [1, 0], [0.9, 0.1], [0, 1]],
        )
        result = m.nearest("apple", k=3)
        assert [s for s, _ in result] == ["fruit", "pear", "car"]
        sims = [score for _, score in result]
        assert sims == sorted(sims, reverse=True)

    def test_excludes_query_itself(self):
        m = _matrix(["a", "b"], [[1, 0], [0, 1]])
        result = m.nearest("a", k=2)
        assert [s for s, _ in result] == ["b"]

    def test_explicit_exclude(self):
        m = _matrix(["a", "b", "c"], [[1, 0], [0.9, 0.1], [0.5, 0.5]])
        result = m.nearest("a", k=2, exclude=["b"])
        assert [s for s, _ in result] == ["c"]

    def test_accepts_vector_query(self):
        m = _matrix(["a", "b"], [[1, 0], [0, 1]])
        # vector pointing at 'a' direction (unnormalized)
        result = m.nearest(np.array([5.0, 0.0], dtype=np.float32), k=2)
        assert [s for s, _ in result] == ["a", "b"]

    def test_vector_query_does_not_self_exclude(self):
        m = _matrix(["a", "b"], [[1, 0], [0, 1]])
        result = m.nearest(np.array([1.0, 0.0], dtype=np.float32), k=1)
        assert result[0][0] == "a"

    def test_zero_vector_query_raises(self):
        m = _matrix(["a"], [[1, 0]])
        with pytest.raises(ValueError, match="zero norm"):
            m.nearest(np.zeros(2, dtype=np.float32))

    def test_unknown_string_query_raises(self):
        m = _matrix(["a"], [[1, 0]])
        with pytest.raises(KeyError):
            m.nearest("missing")


class TestRoundTrip:
    def test_save_and_load_preserves_everything(self, tmp_path):
        m = _matrix(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]], provider_id="prov-x")
        path = tmp_path / "m.npz"
        m.save(path)

        loaded = EmbeddingMatrix.load(path)
        assert loaded.surfaces == m.surfaces
        assert loaded.provider_id == "prov-x"
        assert loaded.vocab_cache_key == "abc"
        np.testing.assert_allclose(loaded.vectors, m.vectors, atol=1e-6)
        # Nearest behavior reproduces.
        assert loaded.nearest("a", k=1)[0][0] == m.nearest("a", k=1)[0][0]

    def test_load_restores_metadata(self, tmp_path):
        m = _matrix(
            ["a"], [[1, 0]], provider_id="provider-42", vocab_cache_key="vk-99"
        )
        path = tmp_path / "m.npz"
        m.save(path)

        with np.load(path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata"]))
        assert metadata["provider_id"] == "provider-42"
        assert metadata["vocab_cache_key"] == "vk-99"
        assert metadata["dim"] == 2
