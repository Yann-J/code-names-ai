"""Tests for the LLM-primary guess scorer (prompt build, parse, retry, schema)."""

from __future__ import annotations

import json

import pytest

from codenames_ai.agent.llm_guess_scorer import (
    LLMGuessScorer,
    ScorerConfig,
    _per_word_schema,
    build_compressed_history,
    build_user_prompt,
    parse_llm_scores,
)
from codenames_ai.game.models import Board, Card, Clue, Color, GuesserView
from codenames_ai.game.state import TurnEvent
from codenames_ai.llm.provider import ChatMessage, LLMProvider


class FakeLLM(LLMProvider):
    """Cycles through pre-canned responses; records every chat call."""

    def __init__(self, responses):
        self._responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls: list[dict] = []

    @property
    def provider_id(self) -> str:
        return "fake-llm"

    def chat(self, messages, *, json_mode=False, json_schema=None):
        msgs = [m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages]
        self.calls.append(
            {"messages": msgs, "json_mode": json_mode, "json_schema": json_schema}
        )
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


def _board(words_with_colors: list[tuple[str, Color, bool]]) -> Board:
    full = list(words_with_colors)
    pad = [(f"pad{i}", Color.NEUTRAL, True) for i in range(25 - len(full))]
    cards = full + pad
    if len(cards) != 25:
        raise AssertionError("board not 25")
    return Board(
        cards=tuple(Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in cards),
        first_team=Color.RED,
    )


def _make_view(board: Board) -> GuesserView:
    return GuesserView(board=board, team=Color.RED)


class TestParseLLMScores:
    def test_parses_strict_payload(self):
        text = json.dumps(
            {
                "fit": {"apple": 0.9, "car": 0.1},
                "danger": {"apple": 0.1, "car": 0.4},
                "continue": True,
            }
        )
        parsed = parse_llm_scores(text, ("apple", "car"))
        assert parsed is not None
        assert parsed.fit["apple"] == pytest.approx(0.9)
        assert parsed.danger["car"] == pytest.approx(0.4)
        assert parsed.continue_flag is True

    def test_clamps_out_of_range(self):
        text = json.dumps(
            {
                "fit": {"a": 1.5, "b": -0.5},
                "danger": {"a": 0.0, "b": 0.0},
                "continue": False,
            }
        )
        parsed = parse_llm_scores(text, ("a", "b"))
        assert parsed is not None
        assert parsed.fit["a"] == 1.0
        assert parsed.fit["b"] == 0.0

    def test_missing_key_returns_none(self):
        text = json.dumps({"fit": {"a": 0.5}, "continue": True})
        assert parse_llm_scores(text, ("a",)) is None

    def test_missing_word_in_map_returns_none(self):
        text = json.dumps(
            {"fit": {"a": 0.5}, "danger": {"a": 0.5}, "continue": True}
        )
        # Expecting two words but only one in maps → invalid.
        assert parse_llm_scores(text, ("a", "b")) is None

    def test_non_bool_continue_returns_none(self):
        text = json.dumps(
            {"fit": {"a": 0.5}, "danger": {"a": 0.5}, "continue": "yes"}
        )
        assert parse_llm_scores(text, ("a",)) is None

    def test_non_dict_payload_returns_none(self):
        assert parse_llm_scores("[]", ("a",)) is None

    def test_extracts_json_from_prose(self):
        text = (
            'Here you go: {"fit": {"a": 0.4}, "danger": {"a": 0.1}, '
            '"continue": true} Thanks!'
        )
        parsed = parse_llm_scores(text, ("a",))
        assert parsed is not None
        assert parsed.continue_flag is True

    def test_unparseable_returns_none(self):
        assert parse_llm_scores("not json at all", ("a",)) is None

    def test_extra_keys_ignored(self):
        text = json.dumps(
            {
                "fit": {"a": 0.5},
                "danger": {"a": 0.2},
                "continue": False,
                "thoughts": "irrelevant",
            }
        )
        parsed = parse_llm_scores(text, ("a",))
        assert parsed is not None


class TestSchema:
    def test_schema_lists_all_words_required(self):
        schema = _per_word_schema(("apple", "banana", "car"))
        s = schema["schema"]
        assert s["required"] == ["fit", "danger", "continue"]
        assert s["properties"]["fit"]["required"] == ["apple", "banana", "car"]
        assert s["properties"]["danger"]["required"] == ["apple", "banana", "car"]


class TestCompressedHistory:
    def test_omits_pass_clues(self):
        events = (
            TurnEvent(team=Color.RED, kind="CLUE", clue=Clue("fruit", 2)),
            TurnEvent(team=Color.RED, kind="GUESS", guess="apple", outcome_color=Color.RED),
            TurnEvent(team=Color.RED, kind="GUESS", guess="car", outcome_color=Color.NEUTRAL),
            TurnEvent(team=Color.BLUE, kind="CLUE", clue=Clue(word="", count=0)),
            TurnEvent(team=Color.RED, kind="CLUE", clue=Clue("vehicle", 1)),
        )
        rows = build_compressed_history(events, omit_current_clue=True)
        assert len(rows) == 1
        assert rows[0].clue_word == "fruit"
        assert rows[0].correct_hits == 1

    def test_counts_correct_hits_only_for_clue_giving_team(self):
        events = (
            TurnEvent(team=Color.RED, kind="CLUE", clue=Clue("fruit", 3)),
            TurnEvent(team=Color.RED, kind="GUESS", guess="apple", outcome_color=Color.RED),
            TurnEvent(team=Color.RED, kind="GUESS", guess="banana", outcome_color=Color.RED),
            TurnEvent(team=Color.RED, kind="GUESS", guess="pear", outcome_color=Color.BLUE),
            TurnEvent(team=Color.BLUE, kind="CLUE", clue=Clue("car", 1)),
        )
        rows = build_compressed_history(events, omit_current_clue=True)
        assert rows[0].correct_hits == 2

    def test_can_include_current_clue(self):
        events = (
            TurnEvent(team=Color.RED, kind="CLUE", clue=Clue("fruit", 2)),
        )
        rows_in = build_compressed_history(events, omit_current_clue=False)
        assert len(rows_in) == 1
        rows_out = build_compressed_history(events, omit_current_clue=True)
        assert rows_out == ()


class TestUserPrompt:
    def test_lists_revealed_and_unrevealed_words(self):
        board = _board(
            [
                ("apple", Color.RED, False),
                ("car", Color.BLUE, True),
                ("ass", Color.ASSASSIN, False),
            ]
        )
        prompt = build_user_prompt(view=_make_view(board), clue=Clue("fruit", 2), history=())
        assert "Clue: \"fruit\" for 2 target card(s)" in prompt
        assert "apple [UNREVEALED]" in prompt
        assert "car [REVEALED: BLUE]" in prompt
        # Unrevealed list at bottom must include 'apple' but not the revealed 'car'.
        unrevealed_section = prompt.split("UNREVEALED words to score")[-1]
        assert "apple" in unrevealed_section
        assert "car" not in unrevealed_section


class TestLLMGuessScorerCalls:
    def _board_two_cards(self):
        return _board(
            [
                ("apple", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )

    def test_schema_mode_sends_structured_response_format(self):
        llm = FakeLLM(
            json.dumps(
                {
                    "fit": {"apple": 0.9, "car": 0.1},
                    "danger": {"apple": 0.05, "car": 0.4},
                    "continue": True,
                }
            )
        )
        scorer = LLMGuessScorer(llm, config=ScorerConfig(schema_mode=True))
        scores, env = scorer.score(
            view=_make_view(self._board_two_cards()),
            clue=Clue("fruit", 1),
            history=(),
        )
        assert scores is not None
        assert env.fallback_path == "llm_primary"
        assert env.schema_used is True
        assert llm.calls[0]["json_schema"] is not None

    def test_prompt_only_mode_uses_json_object(self):
        llm = FakeLLM(
            json.dumps(
                {
                    "fit": {"apple": 0.9, "car": 0.1},
                    "danger": {"apple": 0.05, "car": 0.4},
                    "continue": False,
                }
            )
        )
        scorer = LLMGuessScorer(llm, config=ScorerConfig(schema_mode=False))
        scores, env = scorer.score(
            view=_make_view(self._board_two_cards()),
            clue=Clue("fruit", 1),
            history=(),
        )
        assert scores is not None
        assert env.schema_used is False
        assert llm.calls[0]["json_mode"] is True
        assert llm.calls[0]["json_schema"] is None

    def test_retries_once_on_parse_failure(self):
        good = json.dumps(
            {"fit": {"apple": 0.9, "car": 0.1}, "danger": {"apple": 0.05, "car": 0.4}, "continue": True}
        )
        llm = FakeLLM(["not json", good])
        scorer = LLMGuessScorer(llm, config=ScorerConfig(schema_mode=False, retry_count=1))
        scores, env = scorer.score(
            view=_make_view(self._board_two_cards()),
            clue=Clue("fruit", 1),
            history=(),
        )
        assert scores is not None
        assert env.parse_attempts == 2
        assert env.fallback_path == "llm_primary"

    def test_returns_none_after_retry_exhausted(self):
        llm = FakeLLM(["not json", "still not json"])
        scorer = LLMGuessScorer(llm, config=ScorerConfig(schema_mode=False, retry_count=1))
        scores, env = scorer.score(
            view=_make_view(self._board_two_cards()),
            clue=Clue("fruit", 1),
            history=(),
        )
        assert scores is None
        assert env.fallback_path == "llm_parse_fail"

    def test_keep_raw_response(self):
        good = json.dumps(
            {"fit": {"apple": 0.9, "car": 0.1}, "danger": {"apple": 0.05, "car": 0.4}, "continue": False}
        )
        llm = FakeLLM(good)
        scorer = LLMGuessScorer(
            llm,
            config=ScorerConfig(schema_mode=False, retry_count=0, keep_raw_response=True),
        )
        _, env = scorer.score(
            view=_make_view(self._board_two_cards()),
            clue=Clue("fruit", 1),
            history=(),
        )
        assert env.raw_response == good

    def test_schema_downgrades_to_prompt_only_on_parse_fail(self):
        bad = "not json"
        good = json.dumps(
            {"fit": {"apple": 0.9, "car": 0.1}, "danger": {"apple": 0.05, "car": 0.4}, "continue": True}
        )
        llm = FakeLLM([bad, good])
        scorer = LLMGuessScorer(llm, config=ScorerConfig(schema_mode=True, retry_count=1))
        scores, env = scorer.score(
            view=_make_view(self._board_two_cards()),
            clue=Clue("fruit", 1),
            history=(),
        )
        assert scores is not None
        # Second call must have downgraded: no schema attached.
        assert llm.calls[0]["json_schema"] is not None
        assert llm.calls[1]["json_schema"] is None
        assert env.schema_used is False
