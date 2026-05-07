"""Tests for OpenAICompatibleProvider with the OpenAI client mocked.

We don't test the openai SDK itself — just that we drive it correctly and
honour the cache.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from codenames_ai.llm.cache import LLMCache
from codenames_ai.llm.provider import ChatMessage, OpenAICompatibleProvider


def _mock_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("openai.OpenAI")
def test_chat_returns_assistant_text(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("hello")
    mock_openai_cls.return_value = client

    provider = OpenAICompatibleProvider(
        model="m", base_url="u", api_key="k"
    )
    out = provider.chat([ChatMessage(role="user", content="hi")])
    assert out == "hello"
    args, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "m"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] == 0.0
    assert "response_format" not in kwargs


@patch("openai.OpenAI")
def test_json_mode_sets_response_format(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("{}")
    mock_openai_cls.return_value = client

    provider = OpenAICompatibleProvider(model="m", base_url="u", api_key="k")
    provider.chat([{"role": "user", "content": "hi"}], json_mode=True)

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("openai.OpenAI")
def test_cache_hit_skips_api_call(mock_openai_cls, tmp_path):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("first")
    mock_openai_cls.return_value = client

    cache = LLMCache(tmp_path / "llm.sqlite")
    provider = OpenAICompatibleProvider(
        model="m", base_url="u", api_key="k", cache=cache
    )
    msg = [ChatMessage(role="user", content="hi")]
    a = provider.chat(msg)
    b = provider.chat(msg)

    assert a == "first"
    assert b == "first"
    assert client.chat.completions.create.call_count == 1


@patch("openai.OpenAI")
def test_cache_writes_response_on_miss(mock_openai_cls, tmp_path):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("from-api")
    mock_openai_cls.return_value = client

    cache = LLMCache(tmp_path / "llm.sqlite")
    provider = OpenAICompatibleProvider(
        model="m", base_url="u", api_key="k", cache=cache
    )
    provider.chat([ChatMessage(role="user", content="hi")])

    cached = cache.get(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        base_url="u",
        temperature=0.0,
        json_mode=False,
    )
    assert cached == "from-api"


@patch("openai.OpenAI")
def test_provider_id_combines_base_url_and_model(mock_openai_cls):
    mock_openai_cls.return_value = MagicMock()
    provider = OpenAICompatibleProvider(
        model="my-model", base_url="https://x.example/v1", api_key="k"
    )
    assert provider.provider_id == "https://x.example/v1::my-model"


@patch("openai.OpenAI")
def test_dict_messages_accepted(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("ok")
    mock_openai_cls.return_value = client

    provider = OpenAICompatibleProvider(model="m", base_url="u", api_key="k")
    provider.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
