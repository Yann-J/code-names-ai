from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codenames_ai.web.app import STATIC_PWA_DIR, create_app


def test_pwa_index_when_built():
    if not STATIC_PWA_DIR.is_dir():
        pytest.skip("PWA not built (run: cd web-ui && npm run build)")
    index = STATIC_PWA_DIR / "index.html"
    if not index.is_file():
        pytest.skip("PWA index missing")
    client = TestClient(create_app())
    r = client.get("/app/")
    assert r.status_code == 200
    assert "Word Guess AI" in r.text


def test_pwa_unknown_route_serves_spa_shell():
    if not Path(STATIC_PWA_DIR, "index.html").is_file():
        pytest.skip("PWA not built")
    client = TestClient(create_app())
    r = client.get("/app/play/fake-id-for-spa-fallback")
    assert r.status_code == 200
    assert "root" in r.text or "Word Guess" in r.text
