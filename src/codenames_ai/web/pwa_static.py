from __future__ import annotations

from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

STATIC_PWA_DIR = Path(__file__).resolve().parent / "static" / "pwa"

_ASSET_PREFIX = "assets/"
_ASSET_SUFFIXES = (".js", ".css", ".woff", ".woff2", ".png", ".svg", ".ico", ".webmanifest")


class PwaStaticFiles(StaticFiles):
    """Serve the Vite PWA build with SPA fallback and production cache headers."""

    @staticmethod
    def _is_asset_path(path: str) -> bool:
        rel = path.lstrip("/")
        return rel.startswith(_ASSET_PREFIX) or rel.endswith(_ASSET_SUFFIXES)

    @staticmethod
    def _cache_headers_for(path: str) -> dict[str, str]:
        rel = path.lstrip("/")
        if rel in ("", "index.html"):
            return {"Cache-Control": "no-store, max-age=0"}
        if rel.startswith(_ASSET_PREFIX):
            return {"Cache-Control": "public, max-age=31536000, immutable"}
        return {}

    async def get_response(self, path: str, scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or not self.html or self._is_asset_path(path):
                raise
            response = await super().get_response("index.html", scope)
            return self._with_cache_headers(response, "index.html")

        return self._with_cache_headers(response, path)

    def _with_cache_headers(self, response: Response, path: str) -> Response:
        if response.status_code != 200:
            return response
        for key, value in self._cache_headers_for(path).items():
            response.headers[key] = value
        return response


def mount_pwa(app, *, directory: Path = STATIC_PWA_DIR) -> bool:
    """Mount hashed assets + SPA shell at ``/app``. Returns False if the build is absent."""
    if not directory.is_dir():
        return False
    app.mount("/app", PwaStaticFiles(directory=directory, html=True), name="pwa")
    return True
