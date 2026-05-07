from fastapi.testclient import TestClient

from codenames_ai.web.app import create_app


def test_index_ok():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "Code Names AI" in r.text
