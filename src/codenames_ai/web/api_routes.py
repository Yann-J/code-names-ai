from __future__ import annotations

import secrets
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from codenames_ai.agent.interfaces import NoLegalClueError
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.trace import SpymasterTrace
from codenames_ai.cli.runtime import EvalRuntime
from codenames_ai.game.board import generate_board
from codenames_ai.game.human import HumanGuesser, HumanSpymaster, trivial_spymaster_trace
from codenames_ai.game.models import Color, SpymasterView
from codenames_ai.game.orchestrator import Game
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
    make_players,
    role_key,
)
from codenames_ai.web.play_session import PlaySession, Role
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
def api_create_game(
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
    humans: dict[str, HumanSpymaster | HumanGuesser] = {}
    rs, rg, bs, bg = make_players(rt, roles, humans)
    board = generate_board(rt.game_vocab, seed=body.seed)
    game = Game(
        board,
        red_spymaster=rs,
        red_guesser=rg,
        blue_spymaster=bs,
        blue_guesser=bg,
        seed=body.seed,
    )
    sid = secrets.token_urlsafe(8)
    sess = PlaySession(id=sid, game=game, roles=roles, humans=humans, risk=body.risk)
    sessions.set(sid, sess)
    advance_ai(sess)
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
def api_spymaster(
    sid: str,
    body: SpymasterGuessBody,
    sessions: StoreDep,
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
    trace = trivial_spymaster_trace(w, targets=(), n=body.count)
    hp.prepare(trace)
    sess.game.step()
    advance_ai(sess)
    return build_game_snapshot(sess, include_secret_colors=include_secret_colors)


@api_router.post("/games/{sid}/guesses", response_model=GameSnapshot)
def api_guesses(
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
    advance_ai(sess)
    sess.ui_guess_flash = None
    return build_game_snapshot(sess, include_secret_colors=include_secret_colors, guess_flash=flash)


@api_router.post("/games/{sid}/end-guess-turn", response_model=GameSnapshot)
def api_end_guess_turn(
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
    advance_ai(sess)
    return build_game_snapshot(sess, include_secret_colors=include_secret_colors)


@api_router.post("/analysis", response_model=AnalysisResponse)
def api_analysis(body: AnalysisRequestBody, get_runtime: RuntimeDep) -> AnalysisResponse:
    def _compute_trace(team: Color) -> SpymasterTrace:
        try:
            return rt.spymaster.give_clue(SpymasterView(board=board, team=team))
        except NoLegalClueError:
            return SpymasterTrace(
                chosen=None,
                top_candidates=(),
                weights=ScoringWeights.from_risk(body.risk),
                veto_count=0,
                illegal_count=0,
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
