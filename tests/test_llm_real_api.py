"""End-to-end smoke against a real OpenAI-compatible endpoint.

Skipped unless `LLM_MODEL`, `LLM_API`, and `LLM_KEY` are all set in the
environment. With Ollama running locally that's:

    LLM_MODEL=llama3.2
    LLM_API=http://localhost:11434/v1
    LLM_KEY=ollama

The test sends a tiny prompt and asserts the response is non-empty.
"""

from __future__ import annotations

import os

import pytest

from codenames_ai.llm.provider import ChatMessage, OpenAICompatibleProvider


@pytest.fixture(scope="module")
def real_provider() -> OpenAICompatibleProvider:
    model = os.environ.get("LLM_MODEL")
    base_url = os.environ.get("LLM_API")
    api_key = os.environ.get("LLM_KEY")
    if not (model and base_url and api_key):
        pytest.skip("LLM_MODEL/LLM_API/LLM_KEY not all set")
    return OpenAICompatibleProvider(
        model=model, base_url=base_url, api_key=api_key, timeout=30.0
    )


def test_real_chat_returns_text(real_provider):
    text = real_provider.chat(
        [
            ChatMessage(role="system", content="Reply with the single word OK."),
            ChatMessage(role="user", content="Confirm."),
        ]
    )
    assert isinstance(text, str)
    assert text.strip()
