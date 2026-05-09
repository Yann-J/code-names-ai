from __future__ import annotations

import secrets
from dataclasses import replace
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from codenames_ai.agent.interfaces import NoLegalClueError
from codenames_ai.agent.risk_context import risk_snapshot_for_board
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.trace import SpymasterTrace
from codenames_ai.cli.runtime import EvalRuntime
from codenames_ai.game.board import generate_board
from codenames_ai.game.human import HumanGuesser, HumanSpymaster, trivial_spymaster_trace
from codenames_ai.game.models import Color, SpymasterView
from codenames_ai.web.api_schemas import (
    AnalysisBoardCard,
    AnalysisRequestBody,
    AnalysisResponse,
    CreateGameBody,
    CreateGameResponse,
    GameSnapshot,
    GuessesBody,
    SpymasterGuessBody,
    build_game_snapshot,
    spymaster_trace_to_payload,
)
from codenames_ai.web.game_service import (
    ALLOWED_GUESS_FLASH_KINDS,
    advance_ai,
    apply_human_guess_words,
    ensure_human_clue_legal,
    new_play_session,
    role_key,
)
from codenames_ai.web.live_notify import notify_live_watchers
from codenames_ai.web.play_session import Role
from codenames_ai.web.session_store import SessionStore

EvalRuntimeFactory = Callable[[float], EvalRuntime]


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_runtime_factory(request: Request) -> EvalRuntimeFactory:
    return request.app.state.get_runtime


StoreDep = Annotated[SessionStore, Depends(get_session_store)]
RuntimeDep = Annotated[EvalRuntimeFactory, Depends(get_runtime_factory)]

api_router = APIRouter(prefix="/api", tags=["api"])


def _vocab_too_small() -> JSONResponse:
    return JSONResponse(
        {"detail": "Game vocabulary too small — build caches (fastText + vocab)."},
        status_code=503,
    )


@api_router.post("/games", response_model=CreateGameResponse)
async def api_create_game(
    request: Request,
    body: CreateGameBody,
    sessions: StoreDep,
    get_runtime: RuntimeDep,
) -> CreateGameResponse:
    rt = get_runtime(body.risk)
    if len(rt.game_vocab) < 25:
        raise HTTPException(status_code=503, detail="Game vocabulary too small — build caches (fastText + vocab).")

    roles: dict[Color, dict[str, Role]] = {
        Color.RED: {"spymaster": body.red_spy, "guesser": body.red_guess},
        Color.BLUE: {"spymaster": body.blue_spy, "guesser": body.blue_guess},
    }
    sid = secrets.token_urlsafe(8)
    sess = new_play_session(session_id=sid, rt=rt, seed=body.seed, risk=body.risk, roles=roles)
    sessions.set(sid, sess)
    await notify_live_watchers(request.app, sess, None)
    snap = build_game_snapshot(sess, include_secret_colors=False)
    return CreateGameResponse(id=sid, state=snap)


@api_router.get("/games/{sid}", response_model=GameSnapshot)
def api_get_game(
    sid: str,
    request: Request,
    sessions: StoreDep,
    include_secret_colors: bool = False,
) -> GameSnapshot:
    sess = sessions.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    flash = sess.ui_guess_flash
    sess.ui_guess_flash = None
    qp = request.query_params
    if qp.get("fx") in ALLOWED_GUESS_FLASH_KINDS and qp.get("gw"):
        flash = {"kind": qp["fx"], "word": qp["gw"].strip().lower()}

    return build_game_snapshot(sess, include_secret_colors=include_secret_colors, guess_flash=flash)


@api_router.post("/games/{sid}/spymaster", response_model=GameSnapshot)
async def api_spymaster(
    request: Request,
    sid: str,
    body: SpymasterGuessBody,
    sessions: StoreDep,
    get_runtime: RuntimeDep,
    include_secret_colors: bool = False,
) -> GameSnapshot:
    sess = sessions.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
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
    await notify_live_watchers(request.app, sess, None)
    return build_game_snapshot(sess, include_secret_colors=include_secret_colors)


@api_router.post("/games/{sid}/guesses", response_model=GameSnapshot)
async def api_guesses(
    request: Request,
    sid: str,
    body: GuessesBody,
    sessions: StoreDep,
    include_secret_colors: bool = False,
) -> GameSnapshot:
    sess = sessions.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    key = role_key(sess.game.state.current_team, False)
    if not isinstance(sess.humans.get(key), HumanGuesser):
        raise HTTPException(status_code=400, detail="Not human guesser turn")
    guesses = [x.strip().lower() for x in body.words if x.strip()]
    if not guesses:
        raise HTTPException(status_code=400, detail="No words submitted")
    flash = apply_human_guess_words(sess, guesses)
    sess.ui_guess_flash = None
    await notify_live_watchers(request.app, sess, flash)
    snap = build_game_snapshot(sess, include_secret_colors=include_secret_colors, guess_flash=flash)
    return snap


@api_router.post("/games/{sid}/advance-ai", response_model=GameSnapshot)
async def api_advance_ai(
    request: Request,
    sid: str,
    sessions: StoreDep,
    include_secret_colors: bool = False,
) -> GameSnapshot:
    sess = sessions.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    advance_ai(sess)
    await notify_live_watchers(request.app, sess, None)
    return build_game_snapshot(sess, include_secret_colors=include_secret_colors, guess_flash=None)


@api_router.post("/games/{sid}/end-guess-turn", response_model=GameSnapshot)
async def api_end_guess_turn(
    request: Request,
    sid: str,
    sessions: StoreDep,
    include_secret_colors: bool = False,
) -> GameSnapshot:
    sess = sessions.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    key = role_key(sess.game.state.current_team, False)
    if not isinstance(sess.humans.get(key), HumanGuesser):
        raise HTTPException(status_code=400, detail="Not human guesser turn")
    try:
        sess.game.end_guessing_turn()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Same contract as ``/live/guess/.../end-guess-turn``: notify subscribers, then the client POSTs ``advance-ai``.
    # Avoids blocking the event loop on ``advance_ai`` so live WebSocket sends complete before AI work runs.
    await notify_live_watchers(request.app, sess, None)
    return build_game_snapshot(sess, include_secret_colors=include_secret_colors)


@api_router.post("/analysis", response_model=AnalysisResponse)
def api_analysis(body: AnalysisRequestBody, get_runtime: RuntimeDep) -> AnalysisResponse:
    def _baseline_spy_weights() -> ScoringWeights:
        spy = rt.spymaster
        bw = getattr(spy, "baseline_weights", None)
        if isinstance(bw, ScoringWeights):
            return bw
        return getattr(spy, "weights")

    def _compute_trace(team: Color) -> SpymasterTrace:
        view = SpymasterView(board=board, team=team)
        try:
            trace = rt.spymaster.give_clue(view)
            if trace.risk_snapshot is None:
                rs = risk_snapshot_for_board(
                    board,
                    team,
                    base_risk=float(body.risk),
                    policy=rt.dynamic_risk_policy,
                )
                trace = replace(trace, risk_snapshot=rs)
            return trace
        except NoLegalClueError:
            snap = risk_snapshot_for_board(
                board,
                team,
                base_risk=float(body.risk),
                policy=rt.dynamic_risk_policy,
            )
            return SpymasterTrace(
                chosen=None,
                top_candidates=(),
                weights=_baseline_spy_weights(),
                veto_count=0,
                illegal_count=0,
                risk_snapshot=snap,
            )

    rt = get_runtime(body.risk)
    if len(rt.game_vocab) < 25:
        raise HTTPException(status_code=503, detail="Game vocabulary too small — build caches first.")
    board = generate_board(rt.game_vocab, seed=body.seed)
    traces_payload = {
        Color.RED.value: spymaster_trace_to_payload(_compute_trace(Color.RED)),
        Color.BLUE.value: spymaster_trace_to_payload(_compute_trace(Color.BLUE)),
    }
    board_payload = [
        AnalysisBoardCard(word=c.word, color=c.color.value.lower()) for c in board.cards
    ]
    return AnalysisResponse(
        seed=body.seed,
        risk=body.risk,
        traces=traces_payload,
        board=board_payload,
        first_team=board.first_team.value,
    )
