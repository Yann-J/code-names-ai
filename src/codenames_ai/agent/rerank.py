from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Iterable

from codenames_ai.agent.trace import Candidate, CandidateGuess
from codenames_ai.game.models import Clue, GuesserView, SpymasterView
from codenames_ai.llm.provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankItem:
    index: int  # 1-based index into the shortlist
    score: float  # 0..1
    reason: str


_SPYMASTER_SYSTEM = (
    "You are evaluating candidate clues for the word game Code Names. "
    "For each candidate, give a score from 0.0 (terrible) to 1.0 (excellent) "
    "reflecting how confidently a typical human guesser on the spymaster's team "
    "would correctly identify the intended target words and only those words. "
    "Penalize clues whose meaning is closer to non-friendly cards (especially the "
    "ASSASSIN), clues with awkward double meanings, and clues whose count claims "
    "more targets than the connection actually supports. Be discriminating — "
    "most candidates should NOT score 1.0.\n\n"
    "Respond with strict JSON of shape:\n"
    '{"scores": [{"index": <int 1-based>, "score": <float 0..1>, "reason": "<short>"}, ...]}'
)


_GUESSER_SYSTEM = (
    "You are evaluating which board cards a Code Names spymaster most likely meant "
    "with a given clue. For each candidate card, give a score from 0.0 (very "
    "unlikely) to 1.0 (very likely) that the spymaster intended this card. "
    "Use the revealed cards' colors as context — they are already played and the "
    "spymaster will not have meant them. Penalize cards whose connection to the "
    "clue is weak or contrived. Be discriminating.\n\n"
    "Respond with strict JSON of shape:\n"
    '{"scores": [{"index": <int 1-based>, "score": <float 0..1>, "reason": "<short>"}, ...]}'
)


def _normalize_minmax(values: Iterable[float]) -> list[float]:
    vs = list(values)
    if not vs:
        return []
    lo = min(vs)
    hi = max(vs)
    if hi == lo:
        return [0.5] * len(vs)
    return [(v - lo) / (hi - lo) for v in vs]


def _parse_response(text: str, expected_count: int) -> dict[int, RerankItem]:
    """Parse the LLM's JSON into `{index → RerankItem}`. Tolerant of partial output.

    Missing entries are simply absent from the returned mapping; callers
    decide how to handle them (the rerankers fall back to embedding-only score
    in that case).
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract the first {...} block — some models wrap JSON in prose.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            logger.warning("LLM rerank response was not parseable as JSON")
            return {}
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("LLM rerank response had unparseable JSON region")
            return {}

    raw = obj.get("scores") or obj.get("ratings") or obj.get("evaluations") or []
    out: dict[int, RerankItem] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
            score = float(entry.get("score"))
        except (TypeError, ValueError):
            continue
        score = max(0.0, min(1.0, score))
        reason = str(entry.get("reason", "")).strip()
        if 1 <= idx <= expected_count:
            out[idx] = RerankItem(index=idx, score=score, reason=reason)
    return out


class SpymasterReranker:
    """LLM-based rerank for the spymaster's top-K embedding candidates.

    The LLM sees the full board (with team colors) and the shortlist as
    `(clue, intended_targets, N)` tuples — but **not** the embedding scores.
    Returns the same candidates with `score = α · normalized_embedding + (1-α) · llm_score`,
    plus `llm_score` and `llm_reason` populated.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        top_k: int = 10,
        blend_alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")
        self.llm = llm
        self.top_k = top_k
        self.blend_alpha = blend_alpha

    def rerank(self, shortlist: list[Candidate], view: SpymasterView) -> list[Candidate]:
        if not shortlist:
            return shortlist
        n = min(len(shortlist), self.top_k)
        head = shortlist[:n]

        messages = [
            ChatMessage(role="system", content=_SPYMASTER_SYSTEM),
            ChatMessage(role="user", content=self._user_prompt(head, view)),
        ]
        response = self.llm.chat(messages, json_mode=True)
        parsed = _parse_response(response, expected_count=n)

        normalized = _normalize_minmax(c.embedding_score for c in head)
        out: list[Candidate] = []
        for i, cand in enumerate(head):
            item = parsed.get(i + 1)
            if item is None:
                # Fall back: keep embedding score, flag missing rerank.
                out.append(cand)
                continue
            blended = (
                self.blend_alpha * normalized[i]
                + (1.0 - self.blend_alpha) * item.score
            )
            out.append(
                replace(
                    cand,
                    score=blended,
                    llm_score=item.score,
                    llm_reason=item.reason,
                )
            )
        return out

    @staticmethod
    def _user_prompt(shortlist: list[Candidate], view: SpymasterView) -> str:
        lines = [f"Your team is {view.team.value}. Active board:"]
        for card in view.board.active():
            lines.append(f"- {card.word} [{card.color.value}]")
        lines.append("")
        lines.append("Candidate clues:")
        for i, c in enumerate(shortlist, start=1):
            lines.append(
                f'{i}. clue="{c.clue}" targets={list(c.targets)} N={c.n}'
            )
        return "\n".join(lines)


class GuesserReranker:
    """LLM-based rerank for the guesser's top `N+3` embedding candidates."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        extra_candidates: int = 3,
        blend_alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")
        self.llm = llm
        self.extra_candidates = extra_candidates
        self.blend_alpha = blend_alpha

    def rerank(
        self,
        shortlist: list[CandidateGuess],
        view: GuesserView,
        clue: Clue,
    ) -> list[CandidateGuess]:
        if not shortlist:
            return shortlist

        n = min(len(shortlist), clue.count + self.extra_candidates)
        head = shortlist[:n]

        messages = [
            ChatMessage(role="system", content=_GUESSER_SYSTEM),
            ChatMessage(role="user", content=self._user_prompt(head, view, clue)),
        ]
        response = self.llm.chat(messages, json_mode=True)
        parsed = _parse_response(response, expected_count=n)

        normalized = _normalize_minmax(c.similarity for c in head)
        out: list[CandidateGuess] = []
        for i, cand in enumerate(head):
            item = parsed.get(i + 1)
            if item is None:
                out.append(cand)
                continue
            blended = (
                self.blend_alpha * normalized[i]
                + (1.0 - self.blend_alpha) * item.score
            )
            out.append(
                replace(
                    cand,
                    score=blended,
                    llm_score=item.score,
                    llm_reason=item.reason,
                )
            )
        # Tail (beyond the shortlist) keeps embedding-only scoring.
        out.extend(shortlist[n:])
        return out

    @staticmethod
    def _user_prompt(
        shortlist: list[CandidateGuess], view: GuesserView, clue: Clue
    ) -> str:
        lines = [
            f'Clue: "{clue.word}" for {clue.count} target card(s).',
            f"Your team is {view.team.value}. Board state:",
        ]
        for card in view.board.cards:
            if card.revealed:
                lines.append(f"- {card.word} [REVEALED: {card.color.value}]")
            else:
                lines.append(f"- {card.word} [UNREVEALED]")
        lines.append("")
        lines.append("Candidate cards to score:")
        for i, c in enumerate(shortlist, start=1):
            lines.append(f'{i}. "{c.word}"')
        return "\n".join(lines)
