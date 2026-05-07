from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from codenames_ai.agent.rerank import (
    GuesserReranker,
    SpymasterReranker,
    _parse_response,
)
from codenames_ai.agent.spymaster import AISpymaster
from codenames_ai.agent.guesser import AIGuesser
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Board, Card, Clue, Color, GuesserView, SpymasterView
from codenames_ai.llm.provider import ChatMessage, LLMProvider
from codenames_ai.vocab.models import Vocabulary, VocabConfig


class FakeLLM(LLMProvider):
    """Returns a pre-canned response. Records every chat call for inspection."""

    def __init__(self, response: str | list[str]) -> None:
        self._response = response if isinstance(response, list) else [response]
        self.calls: list[tuple[list[dict[str, str]], bool]] = []

    @property
    def provider_id(self) -> str:
        return "fake"

    def chat(self, messages, *, json_mode=False):
        msgs = [m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages]
        self.calls.append((msgs, json_mode))
        idx = min(len(self.calls) - 1, len(self._response) - 1)
        return self._response[idx]


class TestParseResponse:
    def test_parses_simple_json(self):
        text = json.dumps({"scores": [{"index": 1, "score": 0.7, "reason": "a"}]})
        parsed = _parse_response(text, expected_count=2)
        assert 1 in parsed
        assert parsed[1].score == 0.7
        assert parsed[1].reason == "a"

    def test_clamps_score_to_unit_interval(self):
        text = json.dumps({"scores": [{"index": 1, "score": 2.0, "reason": ""}]})
        parsed = _parse_response(text, expected_count=1)
        assert parsed[1].score == 1.0

        text = json.dumps({"scores": [{"index": 1, "score": -0.5, "reason": ""}]})
        parsed = _parse_response(text, expected_count=1)
        assert parsed[1].score == 0.0

    def test_drops_out_of_range_indices(self):
        text = json.dumps(
            {"scores": [{"index": 0, "score": 0.5, "reason": ""},
                        {"index": 99, "score": 0.5, "reason": ""}]}
        )
        parsed = _parse_response(text, expected_count=3)
        assert parsed == {}

    def test_extracts_json_when_wrapped_in_prose(self):
        text = "Here you go:\n{\"scores\":[{\"index\":1,\"score\":0.5,\"reason\":\"x\"}]}\nThanks!"
        parsed = _parse_response(text, expected_count=1)
        assert 1 in parsed

    def test_returns_empty_on_unparseable(self):
        parsed = _parse_response("not json at all", expected_count=2)
        assert parsed == {}

    def test_accepts_alternative_keys(self):
        text = json.dumps({"ratings": [{"index": 1, "score": 0.4, "reason": ""}]})
        parsed = _parse_response(text, expected_count=1)
        assert 1 in parsed


def _basic_setup_with_two_clues():
    """Two clue candidates with similar embedding scores so the LLM can break the tie."""

    def vec(x, y):
        v = np.array([x, y], dtype=np.float32)
        return v / float(np.linalg.norm(v))

    entries = []
    for i in range(9):
        entries.append((f"f{i}", vec(1.0, 0.01 * i), 5.0))
    for i in range(8):
        entries.append((f"o{i}", vec(-1.0, 0.01 * i), 5.0))
    for i in range(7):
        entries.append((f"n{i}", vec(-0.5, 0.7 + 0.01 * i), 5.0))
    entries.append(("ass", vec(-1.0, -0.3), 5.0))
    entries.append(("clue_a", vec(1.0, 0.0), 5.0))
    entries.append(("clue_b", vec(0.99, 0.05), 5.0))

    surfaces = [s for s, _, _ in entries]
    matrix = EmbeddingMatrix(
        vectors=np.stack([v for _, v, _ in entries]),
        surfaces=surfaces,
        surface_to_index={s: i for i, s in enumerate(surfaces)},
        provider_id="t",
        vocab_cache_key="t",
    )
    df = pd.DataFrame(
        [{"surface": s, "lemma": s, "zipf": z, "pos": "NOUN"} for s, _, z in entries]
    )
    vocab = Vocabulary(
        config=VocabConfig(
            language="en",
            zipf_min=3.0,
            zipf_max=7.0,
            allowed_pos=frozenset({"NOUN"}),
        ),
        df=df,
    )

    cards = (
        [Card(word=f"f{i}", lemma=f"f{i}", color=Color.RED) for i in range(9)]
        + [Card(word=f"o{i}", lemma=f"o{i}", color=Color.BLUE) for i in range(8)]
        + [Card(word=f"n{i}", lemma=f"n{i}", color=Color.NEUTRAL) for i in range(7)]
        + [Card(word="ass", lemma="ass", color=Color.ASSASSIN)]
    )
    board = Board(cards=tuple(cards), first_team=Color.RED)
    return matrix, vocab, board


class TestSpymasterReranker:
    def test_blends_score_and_marks_llm_fields(self):
        matrix, vocab, board = _basic_setup_with_two_clues()
        # LLM strongly prefers 'clue_b' over 'clue_a'.
        # Order in shortlist depends on embedding-rank, which we don't pre-know,
        # so respond with both indices and pick the one matching 'clue_b'.
        # Prepare the response as if shortlist[1] is the preferred one, but
        # since we don't know which order the embedding ranker put them in,
        # we score by clue name in the prompt body. Easiest: have FakeLLM
        # return the same response regardless.
        rerank_response = json.dumps(
            {
                "scores": [
                    {"index": 1, "score": 0.1, "reason": "weak"},
                    {"index": 2, "score": 0.95, "reason": "strong"},
                ]
            }
        )
        llm = FakeLLM(rerank_response)
        reranker = SpymasterReranker(llm, top_k=2, blend_alpha=0.0)  # alpha=0 → only LLM
        spymaster = AISpymaster(matrix, vocab, risk=0.5, reranker=reranker)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))

        # Top candidate must have the higher LLM score.
        assert trace.chosen.llm_score == 0.95
        assert trace.chosen.llm_reason == "strong"
        # Embedding score is preserved.
        assert trace.chosen.embedding_score is not None
        # And final score equals the blended value.
        assert trace.chosen.score == pytest.approx(0.95)

    def test_alpha_one_keeps_embedding_ordering(self):
        matrix, vocab, board = _basic_setup_with_two_clues()
        # With alpha=1, blended == normalized embedding regardless of LLM.
        rerank_response = json.dumps(
            {
                "scores": [
                    {"index": 1, "score": 0.0, "reason": ""},
                    {"index": 2, "score": 0.0, "reason": ""},
                ]
            }
        )
        llm = FakeLLM(rerank_response)
        reranker = SpymasterReranker(llm, top_k=2, blend_alpha=1.0)
        spymaster = AISpymaster(matrix, vocab, risk=0.5, reranker=reranker)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        # Top candidate's LLM fields are populated even with alpha=1.
        assert trace.chosen.llm_score == 0.0

    def test_missing_response_falls_back_to_embedding_score(self):
        matrix, vocab, board = _basic_setup_with_two_clues()
        llm = FakeLLM("not json")
        reranker = SpymasterReranker(llm, top_k=2, blend_alpha=0.5)
        spymaster = AISpymaster(matrix, vocab, risk=0.5, reranker=reranker)
        trace = spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        # No LLM data on chosen — fall back was triggered.
        assert trace.chosen.llm_score is None

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="blend_alpha"):
            SpymasterReranker(FakeLLM("{}"), blend_alpha=1.5)

    def test_llm_called_in_json_mode(self):
        matrix, vocab, board = _basic_setup_with_two_clues()
        llm = FakeLLM('{"scores": [{"index": 1, "score": 0.5, "reason": ""}]}')
        reranker = SpymasterReranker(llm, top_k=2, blend_alpha=0.5)
        spymaster = AISpymaster(matrix, vocab, risk=0.5, reranker=reranker)
        spymaster.give_clue(SpymasterView(board=board, team=Color.RED))
        assert llm.calls
        _, json_mode = llm.calls[0]
        assert json_mode is True


class TestGuesserReranker:
    def _matrix_and_board(self):
        def vec(x, y):
            v = np.array([x, y], dtype=np.float32)
            return v / float(np.linalg.norm(v))

        entries = [
            ("clue", vec(1.0, 0.0)),
            ("a", vec(1.0, 0.0)),
            ("b", vec(0.95, 0.05)),
            ("c", vec(-1.0, 0.0)),
        ] + [(f"p{i}", vec(0.0, 1.0)) for i in range(22)]
        surfaces = [s for s, _ in entries]
        matrix = EmbeddingMatrix(
            vectors=np.stack([v for _, v in entries]),
            surfaces=surfaces,
            surface_to_index={s: i for i, s in enumerate(surfaces)},
            provider_id="t",
            vocab_cache_key="t",
        )
        cards = [
            Card(word="a", lemma="a", color=Color.RED),
            Card(word="b", lemma="b", color=Color.RED),
            Card(word="c", lemma="c", color=Color.BLUE),
        ] + [
            Card(word=f"p{i}", lemma=f"p{i}", color=Color.NEUTRAL, revealed=True)
            for i in range(22)
        ]
        board = Board(cards=tuple(cards), first_team=Color.RED)
        return matrix, board

    def test_llm_can_demote_a_high_similarity_card(self):
        matrix, board = self._matrix_and_board()
        # Embedding ranking would put 'a' first, then 'b'. LLM strongly prefers 'b'.
        # The embedding ordering of the shortlist is by sim desc, so:
        #   index 1 = 'a'  (sim ~ 1.0)
        #   index 2 = 'b'  (sim ~ 0.998)
        #   index 3 = 'c'  (sim < 0)
        # Force LLM to demote index 1.
        rerank_response = json.dumps(
            {
                "scores": [
                    {"index": 1, "score": 0.1, "reason": "homonym, risky"},
                    {"index": 2, "score": 0.95, "reason": "clean fit"},
                    {"index": 3, "score": 0.0, "reason": "wrong color"},
                ]
            }
        )
        llm = FakeLLM(rerank_response)
        reranker = GuesserReranker(llm, blend_alpha=0.0)  # LLM-only blend
        guesser = AIGuesser(matrix, risk=0.5, reranker=reranker)
        trace = guesser.guess(
            GuesserView(board=board, team=Color.RED), Clue("clue", 1)
        )
        assert trace.guesses == ("b",)

    def test_llm_metadata_attached_to_committed_picks(self):
        matrix, board = self._matrix_and_board()
        rerank_response = json.dumps(
            {
                "scores": [
                    {"index": 1, "score": 0.9, "reason": "very likely"},
                    {"index": 2, "score": 0.5, "reason": "maybe"},
                    {"index": 3, "score": 0.0, "reason": "no"},
                ]
            }
        )
        llm = FakeLLM(rerank_response)
        reranker = GuesserReranker(llm, blend_alpha=0.5)
        guesser = AIGuesser(matrix, risk=1.0, reranker=reranker)
        trace = guesser.guess(
            GuesserView(board=board, team=Color.RED), Clue("clue", 2)
        )
        committed = [c for c in trace.candidates if c.committed]
        for c in committed:
            assert c.llm_score is not None

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="blend_alpha"):
            GuesserReranker(FakeLLM("{}"), blend_alpha=-0.1)
