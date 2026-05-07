from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from codenames_ai.llm.cache import LLMCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMProvider(ABC):
    """Generic chat-completion interface.

    Concrete providers wrap whatever API the underlying model exposes; the
    `OpenAICompatibleProvider` covers OpenAI, Mistral, Ollama, vLLM, LM Studio,
    and any other server speaking the OpenAI chat-completions schema.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable identifier for cache keys and trace metadata."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        """Send a list of role-tagged messages and return the assistant text."""


class OpenAICompatibleProvider(LLMProvider):
    """`LLMProvider` backed by the OpenAI Python SDK.

    Set `base_url` to anything that speaks the chat-completions API:
      - https://api.openai.com/v1 (default)
      - https://api.mistral.ai/v1
      - http://localhost:11434/v1 (Ollama)
      - http://localhost:8000/v1 (vLLM, LM Studio, ...)

    `cache` (optional) is checked before every API call; on miss, the response
    is written to it. Pass the same `LLMCache` instance to multiple providers
    if you want one shared SQLite file.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        cache: LLMCache | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.base_url = base_url
        self.cache = cache
        self.temperature = temperature

    @property
    def provider_id(self) -> str:
        return f"{self.base_url}::{self.model}"

    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        msgs = [m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages]

        if self.cache is not None:
            cached = self.cache.get(
                messages=msgs,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                json_mode=json_mode,
            )
            if cached is not None:
                return cached

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("LLM request: model=%s base_url=%s", self.model, self.base_url)
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""

        if self.cache is not None:
            self.cache.put(
                messages=msgs,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                json_mode=json_mode,
                response=text,
            )
        return text


def provider_from_config(
    *,
    model: str,
    base_url: str,
    api_key: str,
    cache: LLMCache | None = None,
    temperature: float = 0.0,
) -> OpenAICompatibleProvider:
    """Convenience factory that builds the default OpenAI-compatible provider."""
    return OpenAICompatibleProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        cache=cache,
        temperature=temperature,
    )
