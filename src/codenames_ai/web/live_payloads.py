from __future__ import annotations

from typing import Literal

from codenames_ai.web.api_schemas import GuessFlash, build_game_snapshot
from codenames_ai.web.live_schemas import LiveSnapshot
from codenames_ai.web.play_session import PlaySession


def live_snapshot_to_json(
    sess: PlaySession,
    role: Literal["guess", "spy"],
    *,
    guess_flash: dict[str, str] | GuessFlash | None = None,
) -> dict:
    include_secret = role == "spy"
    snap = build_game_snapshot(sess, include_secret_colors=include_secret, guess_flash=guess_flash)
    return LiveSnapshot(version=1, role=role, state=snap).model_dump(mode="json")
