"""Single-call LLM scorer for the LLM-primary guesser.

Encapsulates everything provider-facing for one physical guess: prompt
construction (system + user with current clue, full board, compressed history
for both teams), schema-first vs prompt-only JSON selection, ``temperature 0``
decoding, single retry on parse/validation failure, and parsing into a
normalised ``LLMScores`` payload (``fit`` / ``danger`` per unrevealed word
plus the model's ``continue`` flag).

Produces ``ScoredGuess`` for the policy engine and a ``StepEnvelope`` of
debug metadata (model id, schema vs fallback, raw response hash) for traces.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from codenames_ai.game.models import Clue, GuesserView
from codenames_ai.game.state import TurnEvent
from codenames_ai.llm.provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert Code Names guesser. Given the current clue, your team, "
    "the board (with already-revealed colours visible), and a compressed "
    "history of past clue rounds for both teams, score every UNREVEALED word "
    "twice:\n"
    "  - fit ∈ [0, 1]: how strongly the word fits the clue under Codenames "
    "conventions (synonyms, idioms, common collocations, weak associations). "
    "Do NOT score by hidden card colour — colour is unknown for unrevealed "
    "cards.\n"
    "  - danger ∈ [0, 1]: how costly it would be if this guess turned out to "
    "be wrong (assassin-shaped wording, words that an opposing spymaster could "
    "plausibly target, or that a typical neutral picks up). Use only public "
    "board information.\n"
    "Then emit a single boolean continue flag: true to keep guessing after "
    "this commit, false to tap out (conservative human-style stop).\n\n"
    "Output strict JSON with EXACTLY these keys:\n"
    '  {"fit": {"<WORD>": <0..1>, ...}, '
    '"danger": {"<WORD>": <0..1>, ...}, '
    '"continue": <true|false>}\n'
    "The fit and danger maps must contain ENTRY FOR EVERY UNREVEALED WORD "
    "listed — no omissions, no extras. All values must be numbers in [0, 1]."
)


def _per_word_schema(unrevealed_words: tuple[str, ...]) -> dict:
    """JSON Schema for the fit/danger/continue object with required-key set."""
    word_props = {
        w: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for w in unrevealed_words
    }
    fit_obj = {
        "type": "object",
        "properties": word_props,
        "required": list(unrevealed_words),
        "additionalProperties": False,
    }
    danger_obj = dict(fit_obj)
    return {
        "name": "codenames_guess_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "fit": fit_obj,
                "danger": danger_obj,
                "continue": {"type": "boolean"},
            },
            "required": ["fit", "danger", "continue"],
            "additionalProperties": False,
        },
    }


@dataclass(frozen=True)
class CompressedTurn:
    """One row of the cross-team history shown to the model.

    ``correct_hits`` is the count of own-colour reveals the team produced for
    ``clue_word`` before the round ended (wrong colour / stop / out of guesses
    / game over). PASS turns are intentionally omitted by the builder, per the
    PRD "compressed history".
    """

    team: str
    clue_word: str
    clue_count: int
    correct_hits: int


def build_compressed_history(
    turn_history: tuple[TurnEvent, ...],
    *,
    omit_current_clue: bool = True,
) -> tuple[CompressedTurn, ...]:
    """Collapse the full event log into one row per concluded clue round.

    ``omit_current_clue`` drops the trailing clue event when no guesses follow
    it — i.e. the clue that the LLM is about to play against. PASS clues are
    skipped entirely.
    """
    out: list[CompressedTurn] = []
    pending_clue: TurnEvent | None = None
    pending_hits = 0
    for ev in turn_history:
        if ev.kind == "CLUE":
            if pending_clue is not None and pending_clue.clue is not None:
                if not pending_clue.clue.is_pass():
                    out.append(
                        CompressedTurn(
                            team=pending_clue.team.value,
                            clue_word=pending_clue.clue.word,
                            clue_count=int(pending_clue.clue.count),
                            correct_hits=pending_hits,
                        )
                    )
            pending_clue = ev
            pending_hits = 0
            continue
        if ev.kind == "GUESS" and pending_clue is not None:
            if ev.outcome_color == pending_clue.team:
                pending_hits += 1
    if pending_clue is not None and pending_clue.clue is not None:
        if not pending_clue.clue.is_pass():
            if not omit_current_clue:
                out.append(
                    CompressedTurn(
                        team=pending_clue.team.value,
                        clue_word=pending_clue.clue.word,
                        clue_count=int(pending_clue.clue.count),
                        correct_hits=pending_hits,
                    )
                )
    return tuple(out)


def build_user_prompt(
    *,
    view: GuesserView,
    clue: Clue,
    history: tuple[CompressedTurn, ...],
) -> str:
    lines: list[str] = []
    lines.append(f'Clue: "{clue.word}" for {clue.count} target card(s).')
    lines.append(f"Your team: {view.team.value}.")
    lines.append("")
    lines.append("Board (revealed cards show their flipped colour):")
    for card in view.board.cards:
        if card.revealed:
            lines.append(f"- {card.word} [REVEALED: {card.color.value}]")
        else:
            lines.append(f"- {card.word} [UNREVEALED]")
    lines.append("")
    if history:
        lines.append("Compressed history (PASS rounds omitted):")
        for row in history:
            lines.append(
                f'- {row.team} clue "{row.clue_word}" N={row.clue_count} → '
                f"{row.correct_hits} own-colour hit(s)"
            )
        lines.append("")
    unrevealed = [c.word for c in view.board.active()]
    lines.append("UNREVEALED words to score (must include every entry below):")
    for w in unrevealed:
        lines.append(f"- {w}")
    return "\n".join(lines)


@dataclass(frozen=True)
class LLMScores:
    """Validated parser output for one physical guess decision."""

    fit: dict[str, float]
    danger: dict[str, float]
    continue_flag: bool


def _coerce_unit(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(1.0, f))


def parse_llm_scores(text: str, unrevealed: tuple[str, ...]) -> LLMScores | None:
    """Parse ``{"fit": {...}, "danger": {...}, "continue": bool}`` strictly.

    Returns ``None`` when the JSON cannot be parsed, lacks any of the required
    keys, has a missing word in either map, or carries non-numeric values
    (after best-effort coercion). Extra keys outside the contract are ignored.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    fit_raw = obj.get("fit")
    danger_raw = obj.get("danger")
    cont_raw = obj.get("continue")
    if not isinstance(fit_raw, dict) or not isinstance(danger_raw, dict):
        return None
    if not isinstance(cont_raw, bool):
        return None
    fit: dict[str, float] = {}
    danger: dict[str, float] = {}
    for word in unrevealed:
        fv = _coerce_unit(fit_raw.get(word))
        dv = _coerce_unit(danger_raw.get(word))
        if fv is None or dv is None:
            return None
        fit[word] = fv
        danger[word] = dv
    return LLMScores(fit=fit, danger=danger, continue_flag=bool(cont_raw))


@dataclass(frozen=True)
class StepEnvelope:
    """Light debug metadata co-returned with parsed scores."""

    model_id: str
    schema_used: bool
    raw_response_hash: str
    raw_response: str = ""
    fallback_path: str = "llm_primary"
    parse_attempts: int = 1


@dataclass(frozen=True)
class ScorerConfig:
    schema_mode: bool = True
    """Try ``response_format=json_schema`` first; fall back to prompt-only JSON on parse failure."""

    retry_count: int = 1
    """Re-issue the same temperature-0 call once on parse failure (PRD: fixed at 1 for v1)."""

    keep_raw_response: bool = False
    """When True, retain the raw model output in the trace envelope (heavy)."""


class LLMGuessScorer:
    """Build prompts, call the provider with retry, parse and validate scores."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        config: ScorerConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or ScorerConfig()

    def score(
        self,
        *,
        view: GuesserView,
        clue: Clue,
        history: tuple[CompressedTurn, ...],
    ) -> tuple[LLMScores | None, StepEnvelope]:
        unrevealed = tuple(c.word for c in view.board.active())
        if not unrevealed:
            envelope = StepEnvelope(
                model_id=self.llm.provider_id,
                schema_used=False,
                raw_response_hash="",
                fallback_path="no_unrevealed",
            )
            return None, envelope

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=build_user_prompt(view=view, clue=clue, history=history),
            ),
        ]
        schema = _per_word_schema(unrevealed) if self.config.schema_mode else None
        attempts = 1 + max(0, int(self.config.retry_count))
        last_raw = ""
        schema_used = False
        for attempt in range(attempts):
            try:
                if schema is not None:
                    raw = self.llm.chat(messages, json_schema=schema)
                    schema_used = True
                else:
                    raw = self.llm.chat(messages, json_mode=True)
                    schema_used = False
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM guess scorer call failed (attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                # On exception with schema mode, downgrade to prompt-only JSON for the retry.
                if schema is not None:
                    schema = None
                continue
            last_raw = raw
            parsed = parse_llm_scores(raw, unrevealed)
            if parsed is not None:
                envelope = StepEnvelope(
                    model_id=self.llm.provider_id,
                    schema_used=schema_used,
                    raw_response_hash=_hash(raw),
                    raw_response=raw if self.config.keep_raw_response else "",
                    fallback_path="llm_primary",
                    parse_attempts=attempt + 1,
                )
                return parsed, envelope
            logger.warning(
                "LLM guess scorer parse failed (attempt %d/%d); will %s",
                attempt + 1,
                attempts,
                "retry" if attempt + 1 < attempts else "fall back",
            )
            # On parse failure with schema mode, downgrade for the retry.
            if schema is not None:
                schema = None
        envelope = StepEnvelope(
            model_id=self.llm.provider_id,
            schema_used=schema_used,
            raw_response_hash=_hash(last_raw),
            raw_response=last_raw if self.config.keep_raw_response else "",
            fallback_path="llm_parse_fail",
            parse_attempts=attempts,
        )
        return None, envelope


def _hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
