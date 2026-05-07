from __future__ import annotations

from abc import ABC, abstractmethod

from codenames_ai.web.play_session import PlaySession


class SessionStore(ABC):
    """Pluggable session backend; default is in-process dict."""

    @abstractmethod
    def get(self, sid: str) -> PlaySession | None: ...

    @abstractmethod
    def set(self, sid: str, session: PlaySession) -> None: ...

    @abstractmethod
    def delete(self, sid: str) -> None: ...

    @abstractmethod
    def __contains__(self, sid: str) -> bool: ...


class InMemorySessionStore(SessionStore):
    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[str, PlaySession] = {}

    def get(self, sid: str) -> PlaySession | None:
        return self._data.get(sid)

    def set(self, sid: str, session: PlaySession) -> None:
        self._data[sid] = session

    def delete(self, sid: str) -> None:
        self._data.pop(sid, None)

    def __contains__(self, sid: str) -> bool:
        return sid in self._data
