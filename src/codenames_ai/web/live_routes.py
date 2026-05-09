from __future__ import annotations

import random
import secrets
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

from codenames_ai.cli.runtime import EvalRuntime
from codenames_ai.game.human import HumanGuesser, HumanSpymaster, trivial_spymaster_trace
from codenames_ai.game.models import Color
from codenames_ai.web.api_schemas import GuessesBody, SpymasterGuessBody, build_game_snapshot
from codenames_ai.web.game_service import (
    advance_ai,
    apply_human_guess_words,
    ensure_human_clue_legal,
    new_play_session,
    rematch_play_session,
    role_key,
    roles_have_human_guesser,
    roles_have_human_spymaster,
)
from codenames_ai.web.live_payloads import live_snapshot_to_json
from codenames_ai.web.live_registry import LiveRoomRegistry, LiveRoomState
from codenames_ai.web.live_schemas import (
    CreateLiveRoomBody,
    CreateLiveRoomResponse,
    LiveMutationResponse,
    LiveRematchBody,
    LiveSnapshot,
)
from codenames_ai.web.live_urls import live_room_urls
from codenames_ai.web.play_session import PlaySession, Role
from codenames_ai.web.session_store import SessionStore

EvalRuntimeFactory = Callable[[float], EvalRuntime]


def _vocab_exc() -> None:
    raise HTTPException(status_code=503, detail="Game vocabulary too small — build caches (fastText + vocab).")


def get_live_registry(request: Request) -> LiveRoomRegistry:
    return request.app.state.live_registry


LiveRegistryDep = Annotated[LiveRoomRegistry, Depends(get_live_registry)]


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


SessionsDep = Annotated[SessionStore, Depends(get_session_store)]


def get_runtime_factory(request: Request) -> EvalRuntimeFactory:
    return request.app.state.get_runtime


RuntimeDep = Annotated[EvalRuntimeFactory, Depends(get_runtime_factory)]

live_router = APIRouter(prefix="/live", tags=["live"])


def _need_room_guess(meta_wrong: bool, room: LiveRoomState | None) -> LiveRoomState:
    if meta_wrong:
        raise HTTPException(status_code=403, detail="This action requires the operative link")
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def _need_room_spy(meta_wrong: bool, room: LiveRoomState | None) -> LiveRoomState:
    if meta_wrong:
        raise HTTPException(status_code=403, detail="This action requires the captain link")
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def _load_room_session(sessions: SessionStore, room: LiveRoomState) -> PlaySession:
    sess = sessions.get(room.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


def _guess_mutation_snap(sess: PlaySession, guess_flash: dict[str, str] | None) -> LiveSnapshot:
    return LiveSnapshot(
        version=1,
        role="guess",
        state=build_game_snapshot(sess, include_secret_colors=False, guess_flash=guess_flash),
    )


def _spy_mutation_snap(sess: PlaySession) -> LiveSnapshot:
    return LiveSnapshot(
        version=1,
        role="spy",
        state=build_game_snapshot(sess, include_secret_colors=True, guess_flash=None),
    )


async def _ensure_session_for_live(
    body: CreateLiveRoomBody,
    sessions: SessionStore,
    get_runtime: EvalRuntimeFactory,
) -> PlaySession:
    if body.session_id is not None:
        sess = sessions.get(body.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return sess

    rt = get_runtime(body.risk)
    if len(rt.game_vocab) < 25:
        _vocab_exc()

    roles: dict[Color, dict[str, Role]] = {
        Color.RED: {"spymaster": body.red_spy, "guesser": body.red_guess},
        Color.BLUE: {"spymaster": body.blue_spy, "guesser": body.blue_guess},
    }
    sid = secrets.token_urlsafe(8)
    sess = new_play_session(session_id=sid, rt=rt, seed=body.seed, risk=body.risk, roles=roles)
    sessions.set(sid, sess)
    return sess


@live_router.post("/rooms", response_model=CreateLiveRoomResponse)
async def live_create_room(
    body: CreateLiveRoomBody,
    request: Request,
    registry: LiveRegistryDep,
    sessions: SessionsDep,
    get_runtime: RuntimeDep,
) -> CreateLiveRoomResponse:
    sess = await _ensure_session_for_live(body, sessions, get_runtime)
    hg = roles_have_human_guesser(sess.roles)
    hs = roles_have_human_spymaster(sess.roles)
    if not hg and not hs:
        return CreateLiveRoomResponse(
            room_id=sess.id,
            guesser_url=None,
            spymaster_url=None,
            guesser_websocket_url=None,
            spymaster_websocket_url=None,
        )
    room = await registry.ensure_room_for_session(sess.id)
    gu, su, gw, sw = live_room_urls(
        request,
        guess_token=room.guess_token,
        spy_token=room.spy_token,
        include_guess=hg,
        include_spy=hs,
    )
    return CreateLiveRoomResponse(
        room_id=sess.id,
        guesser_url=gu,
        spymaster_url=su,
        guesser_websocket_url=gw,
        spymaster_websocket_url=sw,
    )


@live_router.post("/guess/{token}/guesses", response_model=LiveMutationResponse)
async def live_post_guesses(
    token: str,
    body: GuessesBody,
    registry: LiveRegistryDep,
    sessions: SessionsDep,
) -> LiveMutationResponse:
    meta, room = await registry.resolve_for_guess(token)
    room = _need_room_guess(meta.wrong_band, room)
    sess = _load_room_session(sessions, room)
    key = role_key(sess.game.state.current_team, False)
    if not isinstance(sess.humans.get(key), HumanGuesser):
        raise HTTPException(status_code=400, detail="Not human guesser turn")
    guesses = [x.strip().lower() for x in body.words if x.strip()]
    if not guesses:
        raise HTTPException(status_code=400, detail="No words submitted")
    flash = apply_human_guess_words(sess, guesses)
    sess.ui_guess_flash = None
    await registry.touch_mutation(sess.id)
    await registry.broadcast_snapshots(sess, guess_flash=flash)
    return LiveMutationResponse(snapshot=_guess_mutation_snap(sess, flash))


@live_router.post("/guess/{token}/advance-ai", response_model=LiveMutationResponse)
async def live_post_advance_ai(
    token: str,
    registry: LiveRegistryDep,
    sessions: SessionsDep,
) -> LiveMutationResponse:
    meta, room = await registry.resolve_for_guess(token)
    room = _need_room_guess(meta.wrong_band, room)
    sess = _load_room_session(sessions, room)
    advance_ai(sess)
    await registry.touch_mutation(sess.id)
    await registry.broadcast_snapshots(sess, guess_flash=None)
    return LiveMutationResponse(snapshot=_guess_mutation_snap(sess, None))


@live_router.post("/guess/{token}/end-guess-turn", response_model=LiveMutationResponse)
async def live_end_guess_turn(
    token: str,
    registry: LiveRegistryDep,
    sessions: SessionsDep,
) -> LiveMutationResponse:
    meta, room = await registry.resolve_for_guess(token)
    room = _need_room_guess(meta.wrong_band, room)
    sess = _load_room_session(sessions, room)
    key = role_key(sess.game.state.current_team, False)
    if not isinstance(sess.humans.get(key), HumanGuesser):
        raise HTTPException(status_code=400, detail="Not human guesser turn")
    try:
        sess.game.end_guessing_turn()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Same contract as ``/guess/{token}/guesses``: broadcast end-turn, then the client POSTs ``advance-ai``
    # so the event loop is not blocked by a long ``advance_ai`` chain (other tabs' WS I/O can proceed).
    await registry.touch_mutation(sess.id)
    await registry.broadcast_snapshots(sess, guess_flash=None)
    return LiveMutationResponse(snapshot=_guess_mutation_snap(sess, None))


@live_router.post("/spy/{token}/spymaster", response_model=LiveMutationResponse)
async def live_post_spymaster(
    token: str,
    body: SpymasterGuessBody,
    registry: LiveRegistryDep,
    sessions: SessionsDep,
    get_runtime: RuntimeDep,
) -> LiveMutationResponse:
    meta, room = await registry.resolve_for_spy(token)
    room = _need_room_spy(meta.wrong_band, room)
    sess = _load_room_session(sessions, room)
    team = sess.game.state.current_team
    key = role_key(team, True)
    hp = sess.humans.get(key)
    if not isinstance(hp, HumanSpymaster):
        raise HTTPException(status_code=400, detail="Not human spymaster turn")
    w = body.word.strip().lower()
    rt = get_runtime(sess.risk)
    try:
        ensure_human_clue_legal(sess, rt.game_vocab, w, body.count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    trace = trivial_spymaster_trace(w, targets=(), n=body.count)
    hp.prepare(trace)
    sess.game.step()
    advance_ai(sess)
    await registry.touch_mutation(sess.id)
    await registry.broadcast_snapshots(sess, guess_flash=None)
    return LiveMutationResponse(snapshot=_spy_mutation_snap(sess))


@live_router.post("/spy/{token}/rematch", response_model=LiveMutationResponse)
async def live_rematch(
    token: str,
    body: LiveRematchBody,
    registry: LiveRegistryDep,
    sessions: SessionsDep,
    get_runtime: RuntimeDep,
) -> LiveMutationResponse:
    meta, room = await registry.resolve_for_spy(token)
    room = _need_room_spy(meta.wrong_band, room)
    sess = _load_room_session(sessions, room)
    rt = get_runtime(sess.risk)
    if len(rt.game_vocab) < 25:
        _vocab_exc()
    new_seed = body.seed if body.seed is not None else random.randint(0, 0x7FFFFFFF)
    rematch_play_session(sess, rt, new_seed)
    await registry.touch_mutation(sess.id)
    await registry.broadcast_snapshots(sess, guess_flash=None)
    return LiveMutationResponse(snapshot=_spy_mutation_snap(sess))


@live_router.websocket("/ws/guess/{token}")
async def live_ws_guess(websocket: WebSocket, token: str) -> None:
    registry: LiveRoomRegistry = websocket.app.state.live_registry
    sessions: SessionStore = websocket.app.state.session_store
    await websocket.accept()
    meta, room = await registry.resolve_for_guess(token)
    if meta.wrong_band:
        await websocket.close(code=4403)
        return
    if room is None:
        await websocket.close(code=4404)
        return
    sess = sessions.get(room.session_id)
    if sess is None:
        await websocket.close(code=4404)
        return
    advance_ai(sess)
    payload = live_snapshot_to_json(sess, "guess", guess_flash=None)
    await websocket.send_json(payload)
    await registry.attach_guess_ws(room.session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await registry.detach_guess_ws(room.session_id, websocket)


@live_router.websocket("/ws/spy/{token}")
async def live_ws_spy(websocket: WebSocket, token: str) -> None:
    registry: LiveRoomRegistry = websocket.app.state.live_registry
    sessions: SessionStore = websocket.app.state.session_store
    await websocket.accept()
    meta, room = await registry.resolve_for_spy(token)
    if meta.wrong_band:
        await websocket.close(code=4403)
        return
    if room is None:
        await websocket.close(code=4404)
        return
    sess = sessions.get(room.session_id)
    if sess is None:
        await websocket.close(code=4404)
        return
    advance_ai(sess)
    payload = live_snapshot_to_json(sess, "spy", guess_flash=None)
    await websocket.send_json(payload)
    await registry.attach_spy_ws(room.session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await registry.detach_spy_ws(room.session_id, websocket)
