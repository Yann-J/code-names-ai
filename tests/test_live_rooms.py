"""Black-box tests for /live remote play (issue #2)."""

import time

import pytest
from fastapi.testclient import TestClient

from codenames_ai.web.app import create_app
from codenames_ai.web.live_registry import LiveRoomRegistry


@pytest.fixture
def client() -> TestClient:
    registry = LiveRoomRegistry()
    return TestClient(create_app(live_registry=registry))


@pytest.fixture
def client_short_ttl() -> TestClient:
    registry = LiveRoomRegistry(idle_ttl_sec=0.05)
    return TestClient(create_app(live_registry=registry))


def _skip_no_vocab(resp):
    if resp.status_code == 503:
        pytest.skip("game vocabulary not built")


def test_live_create_returns_independent_urls(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 42, "risk": 0.5})
    _skip_no_vocab(r)
    assert r.status_code == 200
    data = r.json()
    g = data["guesser_url"].rstrip("/").split("/")[-1]
    s = data["spymaster_url"].rstrip("/").split("/")[-1]
    assert g != s
    assert len(g) >= 16
    assert len(s) >= 16


def test_live_create_all_ai_standalone_returns_no_share_urls(client: TestClient):
    r = client.post(
        "/live/rooms",
        json={
            "seed": 43,
            "risk": 0.5,
            "red_spy": "ai",
            "red_guess": "ai",
            "blue_spy": "ai",
            "blue_guess": "ai",
        },
    )
    _skip_no_vocab(r)
    assert r.status_code == 200
    data = r.json()
    assert data["guesser_url"] is None
    assert data["spymaster_url"] is None


def test_live_create_from_session_gates_urls(client: TestClient):
    cg = client.post(
        "/api/games",
        json={
            "seed": 44,
            "risk": 0.5,
            "red_spy": "human",
            "red_guess": "ai",
            "blue_spy": "ai",
            "blue_guess": "ai",
        },
    )
    _skip_no_vocab(cg)
    sid = cg.json()["id"]
    r = client.post("/live/rooms", json={"session_id": sid})
    assert r.status_code == 200
    data = r.json()
    assert data["guesser_url"] is None
    assert data["spymaster_url"] is not None


def test_live_ws_first_message_guess_has_no_secrets(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 1, "risk": 0.5})
    _skip_no_vocab(r)
    guess_tok = r.json()["guesser_url"].rstrip("/").split("/")[-1]
    with client.websocket_connect(f"/live/ws/guess/{guess_tok}") as ws:
        msg = ws.receive_json()
    assert msg["version"] == 1
    assert msg["role"] == "guess"
    for card in msg["state"]["cards"]:
        if not card["revealed"]:
            assert card.get("secret_color") is None


def test_live_ws_spy_includes_secret_colors(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 2, "risk": 0.5})
    _skip_no_vocab(r)
    spy_tok = r.json()["spymaster_url"].rstrip("/").split("/")[-1]
    with client.websocket_connect(f"/live/ws/spy/{spy_tok}") as ws:
        msg = ws.receive_json()
    assert msg["role"] == "spy"
    unrevealed = [x for x in msg["state"]["cards"] if not x["revealed"]]
    assert unrevealed
    assert all(x.get("secret_color") is not None for x in unrevealed)


def test_live_guess_endpoint_rejects_spy_token(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 3, "risk": 0.5})
    _skip_no_vocab(r)
    spy_tok = r.json()["spymaster_url"].rstrip("/").split("/")[-1]
    r2 = client.post(f"/live/guess/{spy_tok}/guesses", json={"words": ["nope"]})
    assert r2.status_code == 403


def test_live_spy_endpoint_rejects_guess_token(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 4, "risk": 0.5})
    _skip_no_vocab(r)
    guess_tok = r.json()["guesser_url"].rstrip("/").split("/")[-1]
    r2 = client.post(f"/live/spy/{guess_tok}/spymaster", json={"word": "x", "count": 1})
    assert r2.status_code == 403


def test_live_broadcast_after_spy_action(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 5, "risk": 0.5})
    _skip_no_vocab(r)
    spy_tok = r.json()["spymaster_url"].rstrip("/").split("/")[-1]
    guess_tok = r.json()["guesser_url"].rstrip("/").split("/")[-1]

    with (
        client.websocket_connect(f"/live/ws/guess/{guess_tok}") as wg,
        client.websocket_connect(f"/live/ws/spy/{spy_tok}") as ws,
    ):
        wg.receive_json()
        ws.receive_json()
        r_sp = client.post(f"/live/spy/{spy_tok}/spymaster", json={"word": "", "count": 0})
        assert r_sp.status_code == 200
        gg = wg.receive_json()
        ss = ws.receive_json()
        assert gg["state"]["current_phase"] == ss["state"]["current_phase"]
        assert gg["state"]["current_team"] == ss["state"]["current_team"]


def test_live_unknown_token_not_found(client: TestClient):
    r = client.post("/live/guess/not-a-real-token-xxxx/guesses", json={"words": ["a"]})
    assert r.status_code == 404


def test_live_rematch_via_spy_preserves_urls(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 7, "risk": 0.5})
    _skip_no_vocab(r)
    data = r.json()
    spy_tok = data["spymaster_url"].rstrip("/").split("/")[-1]

    rm = client.post(f"/live/spy/{spy_tok}/rematch", json={"seed": 99001})
    assert rm.status_code == 200
    body = rm.json()
    assert body["snapshot"]["state"]["seed"] == 99001


def test_live_idle_eviction_without_mutations(client_short_ttl: TestClient):
    r = client_short_ttl.post("/live/rooms", json={"seed": 8, "risk": 0.5})
    _skip_no_vocab(r)
    gt = r.json()["guesser_url"].rstrip("/").split("/")[-1]
    time.sleep(0.25)
    r2 = client_short_ttl.post(f"/live/guess/{gt}/guesses", json={"words": ["z"]})
    assert r2.status_code == 404


def test_live_referrer_policy_header(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 0, "risk": 0.5})
    _skip_no_vocab(r)
    assert r.headers.get("referrer-policy") == "no-referrer"


def test_live_end_guess_turn_broadcasts_before_advance_ai(client: TestClient):
    """Operative end-turn must notify WS clients immediately; AI chain runs only on ``advance-ai``."""
    r = client.post("/live/rooms", json={"seed": 101, "risk": 0.5})
    _skip_no_vocab(r)
    data = r.json()
    guess_tok = data["guesser_url"].rstrip("/").split("/")[-1]
    spy_tok = data["spymaster_url"].rstrip("/").split("/")[-1]

    with (
        client.websocket_connect(f"/live/ws/guess/{guess_tok}") as wg,
        client.websocket_connect(f"/live/ws/spy/{spy_tok}") as ws,
    ):
        g0 = wg.receive_json()
        s0 = ws.receive_json()
        team = g0["state"]["current_team"].lower()
        team_word = next(
            c["word"].lower()
            for c in s0["state"]["cards"]
            if not c["revealed"] and c.get("secret_color", "").lower() == team
        )
        r_sp = client.post(f"/live/spy/{spy_tok}/spymaster", json={"word": "remoteplaytest", "count": 1})
        assert r_sp.status_code == 200
        wg.receive_json()
        ws.receive_json()

        r_g = client.post(f"/live/guess/{guess_tok}/guesses", json={"words": [team_word]})
        assert r_g.status_code == 200
        wg.receive_json()
        ws.receive_json()

        r_end = client.post(f"/live/guess/{guess_tok}/end-guess-turn")
        assert r_end.status_code == 200
        end_spy = ws.receive_json()
        end_guess = wg.receive_json()
        assert end_spy["state"]["current_phase"] == end_guess["state"]["current_phase"]
        assert end_spy["state"]["current_team"] == end_guess["state"]["current_team"]

        r_adv = client.post(f"/live/guess/{guess_tok}/advance-ai")
        assert r_adv.status_code == 200
        wg.receive_json()
        ws.receive_json()


def test_api_end_guess_turn_broadcasts_before_advance_ai(client: TestClient):
    """Host ``POST /api/games/.../end-guess-turn`` must notify live WS before client calls ``advance-ai``."""
    r = client.post("/live/rooms", json={"seed": 102, "risk": 0.5})
    _skip_no_vocab(r)
    data = r.json()
    sid = data["room_id"]
    guess_tok = data["guesser_url"].rstrip("/").split("/")[-1]
    spy_tok = data["spymaster_url"].rstrip("/").split("/")[-1]

    with (
        client.websocket_connect(f"/live/ws/guess/{guess_tok}") as wg,
        client.websocket_connect(f"/live/ws/spy/{spy_tok}") as ws,
    ):
        g0 = wg.receive_json()
        s0 = ws.receive_json()
        team = g0["state"]["current_team"].lower()
        team_word = next(
            c["word"].lower()
            for c in s0["state"]["cards"]
            if not c["revealed"] and c.get("secret_color", "").lower() == team
        )
        r_sp = client.post(f"/api/games/{sid}/spymaster", json={"word": "apitestclue", "count": 1})
        assert r_sp.status_code == 200
        wg.receive_json()
        ws.receive_json()

        r_g = client.post(f"/api/games/{sid}/guesses", json={"words": [team_word]})
        assert r_g.status_code == 200
        wg.receive_json()
        ws.receive_json()

        r_end = client.post(f"/api/games/{sid}/end-guess-turn")
        assert r_end.status_code == 200
        end_spy = ws.receive_json()
        end_guess = wg.receive_json()
        assert end_spy["state"]["current_phase"] == end_guess["state"]["current_phase"]
        assert end_spy["state"]["current_team"] == end_guess["state"]["current_team"]

        r_adv = client.post(f"/api/games/{sid}/advance-ai")
        assert r_adv.status_code == 200
        wg.receive_json()
        ws.receive_json()


def test_api_updates_live_subscribers(client: TestClient):
    r = client.post("/live/rooms", json={"seed": 9, "risk": 0.5})
    _skip_no_vocab(r)
    sid = r.json()["room_id"]
    gt = r.json()["guesser_url"].rstrip("/").split("/")[-1]
    spy_tok = r.json()["spymaster_url"].rstrip("/").split("/")[-1]

    with client.websocket_connect(f"/live/ws/guess/{gt}") as wg:
        wg.receive_json()
        with client.websocket_connect(f"/live/ws/spy/{spy_tok}") as ws:
            ws.receive_json()
            r_api = client.post(f"/api/games/{sid}/spymaster", json={"word": "", "count": 0})
            assert r_api.status_code == 200
            gg = wg.receive_json()
            assert gg["role"] == "guess"
            ws.receive_json()
