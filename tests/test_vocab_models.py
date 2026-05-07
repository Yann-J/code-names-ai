import pandas as pd
import pytest

from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _config(**overrides) -> VocabConfig:
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


class TestVocabConfigCacheKey:
    def test_stable_for_identical_inputs(self):
        assert _config().cache_key() == _config().cache_key()

    def test_changes_when_language_changes(self):
        assert _config(language="en").cache_key() != _config(language="fr").cache_key()

    def test_changes_when_zipf_min_changes(self):
        assert _config(zipf_min=3.0).cache_key() != _config(zipf_min=4.0).cache_key()

    def test_changes_when_zipf_max_changes(self):
        assert _config(zipf_max=7.0).cache_key() != _config(zipf_max=6.5).cache_key()

    def test_changes_when_allowed_pos_changes(self):
        a = _config(allowed_pos=frozenset({"NOUN"})).cache_key()
        b = _config(allowed_pos=frozenset({"NOUN", "ADJ"})).cache_key()
        assert a != b

    def test_invariant_to_allowed_pos_iteration_order(self):
        a = _config(allowed_pos=frozenset(["NOUN", "ADJ"])).cache_key()
        b = _config(allowed_pos=frozenset(["ADJ", "NOUN"])).cache_key()
        assert a == b

    def test_changes_when_min_length_changes(self):
        assert _config(min_length=3).cache_key() != _config(min_length=4).cache_key()

    def test_changes_when_allow_hyphens_changes(self):
        assert _config(allow_hyphens=True).cache_key() != _config(allow_hyphens=False).cache_key()

    def test_changes_when_exclusions_file_content_changes(self, tmp_path):
        path = tmp_path / "ex.txt"
        path.write_text("apple\n")
        key_a = _config(exclusions_path=path).cache_key()
        path.write_text("apple\nbanana\n")
        key_b = _config(exclusions_path=path).cache_key()
        assert key_a != key_b


class TestVocabularyRoundTrip:
    def test_save_and_load_preserves_dataframe(self, tmp_path):
        config = _config()
        df = pd.DataFrame(
            [
                {"surface": "apple", "lemma": "apple", "zipf": 5.1, "pos": "NOUN"},
                {"surface": "frozen", "lemma": "freeze", "zipf": 4.5, "pos": "ADJ"},
            ]
        )
        vocab = Vocabulary(config=config, df=df)
        path = tmp_path / "vocab.parquet"
        vocab.save(path)

        loaded = Vocabulary.load(config, path)
        pd.testing.assert_frame_equal(loaded.df, df.reset_index(drop=True))

    def test_load_rejects_missing_columns(self, tmp_path):
        path = tmp_path / "broken.parquet"
        pd.DataFrame({"surface": ["apple"]}).to_parquet(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            Vocabulary.load(_config(), path)

    def test_len_reports_row_count(self):
        config = _config()
        df = pd.DataFrame(
            [{"surface": w, "lemma": w, "zipf": 5.0, "pos": "NOUN"} for w in ("a", "b", "c")]
        )
        assert len(Vocabulary(config=config, df=df)) == 3

    def test_surfaces_and_lemmas_accessors(self):
        config = _config()
        df = pd.DataFrame(
            [
                {"surface": "running", "lemma": "run", "zipf": 5.0, "pos": "NOUN"},
                {"surface": "frozen", "lemma": "freeze", "zipf": 4.5, "pos": "ADJ"},
            ]
        )
        vocab = Vocabulary(config=config, df=df)
        assert vocab.surfaces == ["running", "frozen"]
        assert vocab.lemmas == ["run", "freeze"]
