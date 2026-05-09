from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from codenames_ai.web.api_schemas import GameSnapshot


class LiveSnapshot(BaseModel):
    version: Literal[1] = 1
    role: Literal["guess", "spy"]
    state: GameSnapshot


class CreateLiveRoomBody(BaseModel):
    """Attach shareable URLs to an existing ``/api/games`` session, or create a standalone session."""

    session_id: str | None = None
    seed: int = 0
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    red_spy: Literal["human", "ai"] = "human"
    red_guess: Literal["human", "ai"] = "human"
    blue_spy: Literal["human", "ai"] = "human"
    blue_guess: Literal["human", "ai"] = "human"


class CreateLiveRoomResponse(BaseModel):
    room_id: str
    guesser_url: str | None
    spymaster_url: str | None
    guesser_websocket_url: str | None
    spymaster_websocket_url: str | None


class LiveMutationResponse(BaseModel):
    snapshot: LiveSnapshot


class LiveRematchBody(BaseModel):
    seed: int | None = None
