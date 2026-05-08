"""JSON API contract tests (requires game vocabulary / runtime, same as web play)."""

import pytest
from fastapi.testclient import TestClient

from codenames_ai.web.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_api_create_and_get_state(client: TestClient):
    r = client.post(
        "/api/games",
        json={
            "seed": 42,
            "risk": 0.5,
            "red_spy": "human",
            "red_guess": "ai",
            "blue_spy": "ai",
            "blue_guess": "ai",
        },
    )
    if r.status_code == 503:
        pytest.skip("game vocabulary not built")
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert "state" in data
    sid = data["id"]
    st = data["state"]
    assert st["id"] == sid
    assert len(st["cards"]) == 25
    assert st["roles"]["red"]["spymaster"] == "human"
    # Unrevealed cards must not include secret_color by default
    for c in st["cards"]:
        assert "word" in c
        assert "revealed" in c
        if not c["revealed"]:
            assert c.get("secret_color") is None

    r2 = client.get(f"/api/games/{sid}")
    assert r2.status_code == 200
    st2 = r2.json()
    assert st2["current_phase"] in ("SPYMASTER", "GUESSER", "DONE")


def test_api_secret_colors_query(client: TestClient):
    r = client.post(
        "/api/games",
        json={
            "seed": 1,
            "risk": 0.5,
            "red_spy": "ai",
            "red_guess": "ai",
            "blue_spy": "ai",
            "blue_guess": "ai",
        },
    )
    if r.status_code == 503:
        pytest.skip("game vocabulary not built")
    sid = r.json()["id"]
    r2 = client.get(f"/api/games/{sid}", params={"include_secret_colors": "true"})
    assert r2.status_code == 200
    unrevealed = [c for c in r2.json()["cards"] if not c["revealed"]]
    assert unrevealed
    assert all(c.get("secret_color") is not None for c in unrevealed)


def test_api_human_spymaster_then_state(client: TestClient):
    r = client.post(
        "/api/games",
        json={
            "seed": 7,
            "risk": 0.5,
            "red_spy": "human",
            "red_guess": "ai",
            "blue_spy": "human",
            "blue_guess": "ai",
        },
    )
    if r.status_code == 503:
        pytest.skip("game vocabulary not built")
    sid = r.json()["id"]
    assert r.json()["state"]["ui"]["show_spymaster_form"] is True
    before_team = r.json()["state"]["current_team"]

    # Pass clue avoids embedding validation on the AI guesser (word not in matrix).
    r2 = client.post(
        f"/api/games/{sid}/spymaster",
        json={"word": "", "count": 0},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["current_phase"] == "SPYMASTER"
    assert body["current_team"] != before_team
    assert body["latest_clue"] is None


def test_api_analysis(client: TestClient):
    r = client.post("/api/analysis", json={"seed": 0, "risk": 0.5})
    if r.status_code == 503:
        pytest.skip("game vocabulary not built")
    assert r.status_code == 200
    data = r.json()
    assert "traces" in data
    assert "RED" in data["traces"]
    assert "BLUE" in data["traces"]
    assert len(data["board"]) == 25
    assert data["first_team"] in ("RED", "BLUE")
