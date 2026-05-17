from __future__ import annotations

from fastapi import Request


def public_http_base(request: Request) -> str:
    """Public site origin for share links, honoring reverse-proxy headers."""
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    host = host.split(",")[0].strip()
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[
        0
    ].strip()
    return f"{proto}://{host}".rstrip("/")


def public_ws_base(request: Request) -> str:
    """WebSocket origin aligned with :func:`public_http_base`."""
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    host = host.split(",")[0].strip()
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[
        0
    ].strip()
    scheme = "wss" if proto == "https" else "ws"
    return f"{scheme}://{host}"
