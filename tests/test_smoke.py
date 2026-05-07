from pathlib import Path

import codenames_ai
from codenames_ai import Config, StoragePaths


def test_package_importable():
    assert codenames_ai.__version__


def test_config_defaults_to_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("CODENAMES_AI_CACHE_DIR", raising=False)
    config = Config()
    assert config.cache_dir == Path.home() / ".cache" / "codenames_ai"


def test_config_reads_cache_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CODENAMES_AI_CACHE_DIR", str(tmp_path))
    config = Config()
    assert config.cache_dir == tmp_path


def test_config_reads_llm_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_KEY", "sk-test-123")
    config = Config()
    assert config.llm_model == "gpt-4o-mini"
    assert config.llm_api == "https://api.openai.com/v1"
    assert config.llm_key is not None
    assert config.llm_key.get_secret_value() == "sk-test-123"


def test_storage_paths_layout(tmp_path):
    config = Config(cache_dir=tmp_path)
    paths = StoragePaths.from_config(config)
    assert paths.vocab_dir_for("en") == tmp_path / "vocab" / "en"
    assert paths.embed_dir_for("en") == tmp_path / "embed" / "en"
    assert paths.llm_cache_path == tmp_path / "llm.sqlite"


def test_storage_paths_ensure_creates_directories(tmp_path):
    paths = StoragePaths(cache_dir=tmp_path)
    paths.ensure()
    assert paths.vocab_dir.is_dir()
    assert paths.embed_dir.is_dir()
    assert paths.models_dir.is_dir()
    assert paths.evals_dir.is_dir()
