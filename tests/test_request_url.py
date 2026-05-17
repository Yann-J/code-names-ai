from codenames_ai.web.request_url import public_http_base, public_ws_base


class _FakeRequest:
    def __init__(self, headers: dict[str, str], *, scheme: str = "http", netloc: str = "localhost:8000"):
        self.headers = headers
        self.url = type("U", (), {"scheme": scheme, "netloc": netloc})()


def test_public_url_uses_forwarded_headers():
    req = _FakeRequest(
        {
            "host": "internal:8000",
            "x-forwarded-host": "play.example.com",
            "x-forwarded-proto": "https",
        },
        scheme="http",
        netloc="internal:8000",
    )
    assert public_http_base(req) == "https://play.example.com"
    assert public_ws_base(req) == "wss://play.example.com"


def test_public_url_falls_back_to_host():
    req = _FakeRequest({"host": "play.example.com"}, scheme="https", netloc="play.example.com")
    assert public_http_base(req) == "https://play.example.com"
