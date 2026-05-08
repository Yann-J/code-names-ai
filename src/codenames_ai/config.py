from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "codenames_ai"


def _default_env_file() -> str:
    """
    Resolve the repo-local `.env` reliably.

    `pydantic-settings` treats `env_file` as a filesystem path; using an
    absolute path avoids surprises when the process working directory differs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / ".env")


class Config(BaseSettings):
    """Top-level application config.

    Most fields use the `CODENAMES_AI_` env prefix; the LLM-related fields use
    bare names (`LLM_MODEL` / `LLM_API` / `LLM_KEY`) so they're trivial to set
    when swapping between OpenAI / Mistral / Ollama / local servers.
    """

    model_config = SettingsConfigDict(
        env_prefix="CODENAMES_AI_",
        env_file=_default_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    fasttext_path: Path | None = None

    # Generic LLM endpoint — anything that speaks the OpenAI chat-completions
    # API (OpenAI, Mistral, Ollama, vLLM, LM Studio, llama.cpp servers, ...).
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_api: str | None = Field(default=None, alias="LLM_API")
    llm_key: SecretStr | None = Field(default=None, alias="LLM_KEY")
