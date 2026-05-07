from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from codenames_ai.agent.interfaces import NoLegalClueError
from codenames_ai.agent.scoring import ScoringWeights
from codenames_ai.agent.trace import SpymasterTrace
from codenames_ai.cli.eval_config import EvalAgentConfigFile
from codenames_ai.cli.runtime import EvalRuntime, build_eval_runtime
from codenames_ai.config import Config
from codenames_ai.game.board import generate_board
from codenames_ai.game.human import (
    HumanGuesser,
    HumanSpymaster,
    trivial_spymaster_trace,
)
from codenames_ai.game.models import Color, SpymasterView
from codenames_ai.game.orchestrator import Game
from codenames_ai.web.api_routes import api_router
from codenames_ai.web.game_service import (
    ALLOWED_GUESS_FLASH_KINDS,
    advance_ai,
    apply_human_guess_words,
    make_players,
    role_key,
)
from codenames_ai.web.play_session import PlaySession, Role
from codenames_ai.web.session_store import InMemorySessionStore, SessionStore

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_PWA_DIR = Path(__file__).resolve().parent / "static" / "pwa"


def create_app(
    agent_config: EvalAgentConfigFile | None = None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Serve the UI. Optionally pass YAML-loaded `EvalAgentConfigFile` (--config).

    With no agent config: embedding-only agents (``llm_rerank=False``), same as before.
    Form ``risk`` still overrides YAML ``risk`` per session.

    Pass ``session_store`` to swap the in-memory default for a custom
    `SessionStore` (for example Redis-backed) without changing route handlers.
    """

    def cfg_at_risk(risk: float) -> EvalAgentConfigFile:
        if agent_config is None:
            return EvalAgentConfigFile(label="web", llm_rerank=False, risk=risk)
        return agent_config.model_copy(update={"risk": risk})

    @lru_cache(maxsize=32)
    def _get_runtime(risk: float) -> EvalRuntime:
        return build_eval_runtime(cfg_at_risk(risk), Config())

    store = session_store or InMemorySessionStore()

    app = FastAPI(title="Code Names AI", version="0.1.0")
    app.state.session_store = store
    app.state.get_runtime = _get_runtime
    app.include_router(api_router)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def sessions() -> SessionStore:
        return app.state.session_store

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request},
        )

    @app.get("/play", response_class=HTMLResponse)
    def play_form(request: Request):
        return templates.TemplateResponse(
            request,
            "play_new.html",
            {"request": request},
        )

    @app.post("/play/start")
    def play_start(
        request: Request,
        seed: int = Form(0),
        risk: float = Form(0.5),
        red_spy: str = Form("ai"),
        red_guess: str = Form("ai"),
        blue_spy: str = Form("ai"),
        blue_guess: str = Form("ai"),
    ):
        rt = _get_runtime(risk=risk)
        if len(rt.game_vocab) < 25:
            return HTMLResponse(
                "Game vocabulary too small — build caches (fastText + vocab).",
                status_code=503,
            )
        roles: dict[Color, dict[str, Role]] = {
            Color.RED: {"spymaster": red_spy, "guesser": red_guess},  # type: ignore[assignment]
            Color.BLUE: {"spymaster": blue_spy, "guesser": blue_guess},  # type: ignore[assignment]
        }
        humans: dict[str, HumanSpymaster | HumanGuesser] = {}
        rs, rg, bs, bg = make_players(rt, roles, humans)
        board = generate_board(rt.game_vocab, seed=seed)
        game = Game(
            board,
            red_spymaster=rs,
            red_guesser=rg,
            blue_spymaster=bs,
            blue_guesser=bg,
            seed=seed,
        )
        sid = secrets.token_urlsafe(8)
        sess = PlaySession(id=sid, game=game, roles=roles, humans=humans, risk=risk)
        sessions().set(sid, sess)
        advance_ai(sess)
        return RedirectResponse(f"/play/{sid}", status_code=303)

    @app.get("/play/{sid}", response_class=HTMLResponse)
    def play_view(request: Request, sid: str):
        sess = sessions().get(sid)
        if sess is None:
            return RedirectResponse("/play", status_code=303)
        flash = sess.ui_guess_flash
        sess.ui_guess_flash = None
        qp = request.query_params
        if qp.get("fx") in ALLOWED_GUESS_FLASH_KINDS and qp.get("gw"):
            flash = {"kind": qp["fx"], "word": qp["gw"].strip().lower()}
        return templates.TemplateResponse(
            request,
            "play.html",
            {"request": request, "sess": sess, "guess_flash": flash},
        )

    @app.post("/play/{sid}/spymaster", response_class=HTMLResponse)
    def play_spymaster_submit(
        request: Request,
        sid: str,
        word: str = Form(...),
        count: int = Form(...),
    ):
        sess = sessions().get(sid)
        if sess is None:
            return HTMLResponse("Session expired", status_code=404)
        team = sess.game.state.current_team
        key = role_key(team, True)
        hp = sess.humans.get(key)
        if not isinstance(hp, HumanSpymaster):
            return HTMLResponse("Not human spymaster turn", status_code=400)
        w = word.strip().lower()
        trace = trivial_spymaster_trace(w, targets=(), n=count)
        hp.prepare(trace)
        sess.game.step()
        advance_ai(sess)
        return RedirectResponse(f"/play/{sid}", status_code=303)

    @app.post("/play/{sid}/guesser", response_class=HTMLResponse)
    def play_guesser_submit(
        request: Request,
        sid: str,
        words: str = Form(...),
    ):
        sess = sessions().get(sid)
        if sess is None:
            return HTMLResponse("Session expired", status_code=404)
        team = sess.game.state.current_team
        key = role_key(team, False)
        if not isinstance(sess.humans.get(key), HumanGuesser):
            return HTMLResponse("Not human guesser turn", status_code=400)
        guess_list = [x.strip().lower() for x in words.split(",") if x.strip()]
        if not guess_list:
            return HTMLResponse("No word submitted", status_code=400)
        apply_human_guess_words(sess, guess_list)
        advance_ai(sess)
        flash = sess.ui_guess_flash
        sess.ui_guess_flash = None
        if flash and flash.get("kind") in ALLOWED_GUESS_FLASH_KINDS:
            q = urlencode({"fx": flash["kind"], "gw": flash["word"]})
            return RedirectResponse(f"/play/{sid}?{q}", status_code=303)
        return RedirectResponse(f"/play/{sid}", status_code=303)

    @app.post("/play/{sid}/end-guess-turn")
    def play_end_guess_turn(request: Request, sid: str):
        sess = sessions().get(sid)
        if sess is None:
            return HTMLResponse("Session expired", status_code=404)
        team = sess.game.state.current_team
        key = role_key(team, False)
        if not isinstance(sess.humans.get(key), HumanGuesser):
            return HTMLResponse("Not human guesser turn", status_code=400)
        try:
            sess.game.end_guessing_turn()
        except ValueError as e:
            return HTMLResponse(str(e), status_code=400)
        advance_ai(sess)
        return RedirectResponse(f"/play/{sid}", status_code=303)

    @app.get("/analysis", response_class=HTMLResponse)
    def analysis_get(request: Request):
        return templates.TemplateResponse(
            request,
            "analysis.html",
            {"request": request, "trace": None, "board": None},
        )

    @app.post("/analysis", response_class=HTMLResponse)
    def analysis_post(
        request: Request,
        seed: int = Form(0),
        risk: float = Form(0.5),
    ):
        rt = _get_runtime(risk=risk)
        if len(rt.game_vocab) < 25:
            return HTMLResponse(
                "Game vocabulary too small — build caches first.",
                status_code=503,
            )
        board = generate_board(rt.game_vocab, seed=seed)
        try:
            trace = rt.spymaster.give_clue(
                SpymasterView(board=board, team=board.first_team)
            )
        except NoLegalClueError:
            trace = SpymasterTrace(
                chosen=None,
                top_candidates=(),
                weights=ScoringWeights.from_risk(risk),
                veto_count=0,
                illegal_count=0,
            )
        return templates.TemplateResponse(
            request,
            "analysis.html",
            {
                "request": request,
                "trace": trace,
                "board": board,
                "seed": seed,
                "risk": risk,
            },
        )

    if STATIC_PWA_DIR.is_dir():
        root = STATIC_PWA_DIR.resolve()

        def _pwa_file_or_none(rel: str) -> Path | None:
            if not rel or rel.startswith("..") or "/.." in rel:
                return None
            p = (STATIC_PWA_DIR / rel).resolve()
            try:
                p.relative_to(root)
            except ValueError:
                return None
            return p if p.is_file() else None

        @app.get("/app")
        def pwa_index_no_slash():
            return FileResponse(STATIC_PWA_DIR / "index.html")

        @app.get("/app/")
        def pwa_index():
            return FileResponse(STATIC_PWA_DIR / "index.html")

        @app.get("/app/{full_path:path}")
        def pwa_shell_or_asset(full_path: str):
            hit = _pwa_file_or_none(full_path)
            if hit is not None:
                return FileResponse(hit)
            return FileResponse(STATIC_PWA_DIR / "index.html")

    return app
