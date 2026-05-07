from codenames_ai.llm.cache import LLMCache


def test_get_returns_none_on_miss(tmp_path):
    cache = LLMCache(tmp_path / "llm.sqlite")
    assert (
        cache.get(
            messages=[{"role": "user", "content": "x"}],
            model="m",
            base_url="u",
            temperature=0.0,
            json_mode=False,
        )
        is None
    )


def test_put_then_get_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "llm.sqlite")
    args = dict(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        base_url="u",
        temperature=0.0,
        json_mode=False,
    )
    cache.put(**args, response="hello")
    assert cache.get(**args) == "hello"


def test_distinct_keys_per_field(tmp_path):
    cache = LLMCache(tmp_path / "llm.sqlite")
    base = dict(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        base_url="u",
        temperature=0.0,
        json_mode=False,
    )
    cache.put(**base, response="A")

    # Each variant should be a cache miss until written.
    for change in (
        {"messages": [{"role": "user", "content": "y"}]},
        {"model": "n"},
        {"base_url": "v"},
        {"temperature": 0.5},
        {"json_mode": True},
    ):
        variant = {**base, **change}
        assert cache.get(**variant) is None, change


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "llm.sqlite"
    args = dict(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        base_url="u",
        temperature=0.0,
        json_mode=False,
    )
    cache = LLMCache(path)
    cache.put(**args, response="persisted")
    cache.close()

    reopened = LLMCache(path)
    assert reopened.get(**args) == "persisted"


def test_put_replaces_existing_entry(tmp_path):
    cache = LLMCache(tmp_path / "llm.sqlite")
    args = dict(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        base_url="u",
        temperature=0.0,
        json_mode=False,
    )
    cache.put(**args, response="A")
    cache.put(**args, response="B")
    assert cache.get(**args) == "B"
