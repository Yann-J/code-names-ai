from __future__ import annotations

from fastapi import Request

from codenames_ai.web.request_url import public_http_base, public_ws_base


def live_room_urls(
    request: Request,
    *,
    guess_token: str,
    spy_token: str,
    include_guess: bool,
    include_spy: bool,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return ``(guesser_http, spymaster_http, guesser_ws, spymaster_ws)`` omitted when not shareable."""
    base = public_http_base(request)
    ws_base = public_ws_base(request)
    g_http = f"{base}/app/remote/guess/{guess_token}" if include_guess else None
    s_http = f"{base}/app/remote/spy/{spy_token}" if include_spy else None
    g_ws = f"{ws_base}/live/ws/guess/{guess_token}" if include_guess else None
    s_ws = f"{ws_base}/live/ws/spy/{spy_token}" if include_spy else None
    return (g_http, s_http, g_ws, s_ws)
