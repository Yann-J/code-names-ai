from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from codenames_ai.cli.eval_config import EvalAgentConfigFile, RiskConfig, ScoringConfig
from codenames_ai.cli.runtime import EvalRuntime, build_eval_runtime
from codenames_ai.config import Config
from codenames_ai.web.api_routes import api_router
from codenames_ai.web.live_registry import LiveRoomRegistry
from codenames_ai.web.live_routes import live_router
from codenames_ai.web.pwa_static import mount_pwa
from codenames_ai.web.session_store import InMemorySessionStore, SessionStore


def create_app(
    agent_config: EvalAgentConfigFile | None = None,
    session_store: SessionStore | None = None,
    live_registry: LiveRoomRegistry | None = None,
    *,
    include_static: bool = True,
) -> FastAPI:
    """Serve JSON API routes and, optionally, the React PWA shell.

    When ``include_static`` is false (production behind nginx), only API and live
    routes are registered; static assets are served separately.
    """

    def cfg_at_risk(risk: float) -> EvalAgentConfigFile:
        if agent_config is None:
            return EvalAgentConfigFile(
                label="web",
                scoring=ScoringConfig(llm_rerank=False),
                risk=RiskConfig(base_risk=risk),
            )
        return agent_config.model_copy(update={"risk": RiskConfig(base_risk=risk)})

    @lru_cache(maxsize=32)
    def _get_runtime(risk: float) -> EvalRuntime:
        return build_eval_runtime(cfg_at_risk(risk), Config())

    store = session_store or InMemorySessionStore()
    live_reg = live_registry if live_registry is not None else LiveRoomRegistry()

    app = FastAPI(title="Code Names AI", version="0.1.0")
    app.state.session_store = store
    app.state.live_registry = live_reg
    app.state.get_runtime = _get_runtime
    app.include_router(api_router)
    app.include_router(live_router)

    @app.middleware("http")
    async def live_referrer_policy(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/live"):
            response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    if include_static and mount_pwa(app):

        @app.get("/")
        def root_index():
            return RedirectResponse(url="/app/")

        @app.get("/app")
        def pwa_index_no_slash():
            return RedirectResponse(url="/app/")

    return app
