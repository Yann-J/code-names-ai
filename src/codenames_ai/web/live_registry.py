from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from starlette.websockets import WebSocket

from codenames_ai.web.play_session import PlaySession


@dataclass(frozen=True)
class LiveTokenBandResolve:
    wrong_band: bool = False


@dataclass
class LiveRoomState:
    session_id: str
    guess_token: str
    spy_token: str
    last_mutation_ts: float
    ws_guess: list[WebSocket] = field(default_factory=list)
    ws_spy: list[WebSocket] = field(default_factory=list)


class LiveRoomRegistry:
    """In-memory live sessions keyed by persisted game ``session_id`` (single-process v1)."""

    def __init__(self, idle_ttl_sec: float = 86400.0) -> None:
        self._lock = asyncio.Lock()
        self._by_session: dict[str, LiveRoomState] = {}
        self._guess_to_session: dict[str, str] = {}
        self._spy_to_session: dict[str, str] = {}
        self._idle_ttl = idle_ttl_sec

    async def _gc_unlocked(self, now: float) -> None:
        expire_before = now - self._idle_ttl
        dead = [sid for sid, r in self._by_session.items() if r.last_mutation_ts < expire_before]
        for sid in dead:
            await self._purge_session_unlocked(sid)

    async def ensure_room_for_session(self, session_id: str) -> LiveRoomState:
        import secrets

        async with self._lock:
            await self._gc_unlocked(time.time())
            existing = self._by_session.get(session_id)
            if existing is not None:
                return existing

            guess_token = secrets.token_urlsafe(16)
            spy_token = secrets.token_urlsafe(16)
            while spy_token == guess_token:
                spy_token = secrets.token_urlsafe(16)

            state = LiveRoomState(
                session_id=session_id,
                guess_token=guess_token,
                spy_token=spy_token,
                last_mutation_ts=time.time(),
            )
            self._by_session[session_id] = state
            self._guess_to_session[guess_token] = session_id
            self._spy_to_session[spy_token] = session_id
            return state

    async def resolve_for_guess(self, token: str) -> tuple[LiveTokenBandResolve, LiveRoomState | None]:
        async with self._lock:
            now = time.time()
            await self._gc_unlocked(now)
            if token in self._spy_to_session:
                return LiveTokenBandResolve(wrong_band=True), None
            sid = self._guess_to_session.get(token)
            if sid is None:
                return LiveTokenBandResolve(), None
            room = self._by_session.get(sid)
            if room is None:
                return LiveTokenBandResolve(), None
            return LiveTokenBandResolve(), room

    async def resolve_for_spy(self, token: str) -> tuple[LiveTokenBandResolve, LiveRoomState | None]:
        async with self._lock:
            now = time.time()
            await self._gc_unlocked(now)
            if token in self._guess_to_session:
                return LiveTokenBandResolve(wrong_band=True), None
            sid = self._spy_to_session.get(token)
            if sid is None:
                return LiveTokenBandResolve(), None
            room = self._by_session.get(sid)
            if room is None:
                return LiveTokenBandResolve(), None
            return LiveTokenBandResolve(), room

    async def attach_guess_ws(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._by_session.get(session_id)
            if room is not None:
                room.ws_guess.append(ws)

    async def attach_spy_ws(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._by_session.get(session_id)
            if room is not None:
                room.ws_spy.append(ws)

    async def detach_guess_ws(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._by_session.get(session_id)
            if room is not None and ws in room.ws_guess:
                room.ws_guess.remove(ws)

    async def detach_spy_ws(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._by_session.get(session_id)
            if room is not None and ws in room.ws_spy:
                room.ws_spy.remove(ws)

    async def touch_mutation(self, session_id: str) -> None:
        async with self._lock:
            room = self._by_session.get(session_id)
            if room is not None:
                room.last_mutation_ts = time.time()

    async def broadcast_snapshots(
        self,
        sess: PlaySession,
        *,
        guess_flash: dict[str, str] | None = None,
    ) -> None:
        from codenames_ai.web.live_payloads import live_snapshot_to_json

        async with self._lock:
            room = self._by_session.get(sess.id)
            if room is None:
                return
            sess.live_mutation_seq += 1
            guess_body = live_snapshot_to_json(sess, "guess", guess_flash=guess_flash)
            spy_body = live_snapshot_to_json(sess, "spy", guess_flash=None)
            g_clients = list(room.ws_guess)
            s_clients = list(room.ws_spy)
        for ws in g_clients:
            try:
                await ws.send_json(guess_body)
            except Exception:
                pass
        for ws in s_clients:
            try:
                await ws.send_json(spy_body)
            except Exception:
                pass

    async def _purge_session_unlocked(self, session_id: str) -> None:
        room = self._by_session.pop(session_id, None)
        if room is None:
            return
        self._guess_to_session.pop(room.guess_token, None)
        self._spy_to_session.pop(room.spy_token, None)
        for ws in list(room.ws_guess) + list(room.ws_spy):
            try:
                await ws.close(code=4408)
            except Exception:
                pass
