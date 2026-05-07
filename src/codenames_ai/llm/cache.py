from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class LLMCache:
    """SQLite-backed cache for LLM responses.

    Keyed by a deterministic hash of `(messages, model, base_url, temperature,
    json_mode)` so the same logical request to the same endpoint hits the
    cache. Designed to be shared across multiple `LLMProvider` instances —
    swap providers, share the cache file.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                base_url TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        base_url: str,
        temperature: float,
        json_mode: bool,
    ) -> str | None:
        key = self._key(messages, model, base_url, temperature, json_mode)
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        base_url: str,
        temperature: float,
        json_mode: bool,
        response: str,
    ) -> None:
        key = self._key(messages, model, base_url, temperature, json_mode)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses(key, model, base_url, response, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, model, base_url, response, time.time()),
            )
            self._conn.commit()

    @staticmethod
    def _key(
        messages: list[dict[str, Any]],
        model: str,
        base_url: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        payload = json.dumps(
            {
                "messages": messages,
                "model": model,
                "base_url": base_url,
                "temperature": round(float(temperature), 6),
                "json_mode": bool(json_mode),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
