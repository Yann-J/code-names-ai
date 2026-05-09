from __future__ import annotations

from typing import Any, TYPE_CHECKING

from codenames_ai.web.play_session import PlaySession

if TYPE_CHECKING:
    from codenames_ai.web.live_registry import LiveRoomRegistry


async def notify_live_watchers(
    app: Any,
    sess: PlaySession,
    guess_flash: dict[str, str] | None,
) -> None:
    reg: LiveRoomRegistry | None = getattr(app.state, "live_registry", None)
    if reg is None:
        return
    await reg.touch_mutation(sess.id)
    await reg.broadcast_snapshots(sess, guess_flash=guess_flash)
