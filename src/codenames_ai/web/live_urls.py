from __future__ import annotations

from fastapi import Request


def live_room_urls(
    request: Request,
    *,
    guess_token: str,
    spy_token: str,
    include_guess: bool,
    include_spy: bool,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return ``(guesser_http, spymaster_http, guesser_ws, spymaster_ws)`` omitted when not shareable."""
    base = str(request.base_url).rstrip("/")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host") or request.url.netloc
    ws_base = f"{scheme}://{host}"
    g_http = f"{base}/app/remote/guess/{guess_token}" if include_guess else None
    s_http = f"{base}/app/remote/spy/{spy_token}" if include_spy else None
    g_ws = f"{ws_base}/live/ws/guess/{guess_token}" if include_guess else None
    s_ws = f"{ws_base}/live/ws/spy/{spy_token}" if include_spy else None
    return (g_http, s_http, g_ws, s_ws)
