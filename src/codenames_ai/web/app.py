from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from codenames_ai.cli.eval_config import EvalAgentConfigFile, ScoringConfig
from codenames_ai.cli.runtime import EvalRuntime, build_eval_runtime
from codenames_ai.config import Config
from codenames_ai.web.api_routes import api_router
from codenames_ai.web.session_store import InMemorySessionStore, SessionStore

STATIC_PWA_DIR = Path(__file__).resolve().parent / "static" / "pwa"


def create_app(
    agent_config: EvalAgentConfigFile | None = None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Serve API routes and the React PWA shell.

    The legacy HTMX/Jinja UI has been removed.
    """

    def cfg_at_risk(risk: float) -> EvalAgentConfigFile:
        if agent_config is None:
            return EvalAgentConfigFile(
                label="web", scoring=ScoringConfig(llm_rerank=False), risk=risk
            )
        return agent_config.model_copy(update={"risk": risk})

    @lru_cache(maxsize=32)
    def _get_runtime(risk: float) -> EvalRuntime:
        return build_eval_runtime(cfg_at_risk(risk), Config())

    store = session_store or InMemorySessionStore()

    app = FastAPI(title="Code Names AI", version="0.1.0")
    app.state.session_store = store
    app.state.get_runtime = _get_runtime
    app.include_router(api_router)

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

        @app.get("/")
        def root_index():
            return RedirectResponse(url="/app")

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
