"""Integration tests for the LLM-primary ``LLMGuesser`` adapter.

Uses ``FakeLLM`` to script per-physical-guess scores so we can assert
end-to-end policy behaviour: argmax commits, gate stops, embedding/uniform
fallbacks, orchestrator plumbing through ``play_turn``.
"""

from __future__ import annotations

import json

import numpy as np

from codenames_ai.agent.llm_guess_policy import ContinueGate
from codenames_ai.agent.llm_guess_scorer import LLMGuessScorer, ScorerConfig
from codenames_ai.agent.llm_guesser import LLMGuesser
from codenames_ai.agent.trace import GuesserTrace
from codenames_ai.embedding.matrix import EmbeddingMatrix
from codenames_ai.game.models import Board, Card, Clue, Color, GuesserView
from codenames_ai.game.orchestrator import Game
from codenames_ai.llm.provider import ChatMessage, LLMProvider


class FakeLLM(LLMProvider):
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


def _payload(fit: dict, danger: dict, continue_flag: bool) -> str:
    return json.dumps({"fit": fit, "danger": danger, "continue": continue_flag})


def _board(words_with_colors: list[tuple[str, Color, bool]]) -> Board:
    full = list(words_with_colors)
    pad_n = 25 - len(full)
    full += [(f"pad{i}", Color.NEUTRAL, True) for i in range(pad_n)]
    return Board(
        cards=tuple(Card(word=w, lemma=w, color=c, revealed=r) for w, c, r in full),
        first_team=Color.RED,
    )


def _scorer(llm: LLMProvider) -> LLMGuessScorer:
    return LLMGuessScorer(llm, config=ScorerConfig(schema_mode=False, retry_count=0))


class TestSingleStepArgmax:
    def test_commits_argmax_of_combined_score(self):
        # apple has highest fit AND lowest danger → wins regardless of λ.
        board = _board(
            [
                ("apple", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM(
            _payload(
                fit={"apple": 0.9, "car": 0.2},
                danger={"apple": 0.05, "car": 0.4},
                continue_flag=False,
            )
        )
        guesser = LLMGuesser(_scorer(llm), lambda_danger=0.5)
        trace = guesser.guess(view, Clue("fruit", 1))
        assert trace.guesses == ("apple",)
        assert trace.llm_steps[0].guess == "apple"
        assert trace.llm_steps[0].fallback_path == "llm_primary"

    def test_lambda_can_flip_argmax(self):
        # fit ranks apple > danger ranks apple — but a high λ punishes its
        # large danger and flips the pick to safe runner-up.
        board = _board(
            [
                ("apple", Color.RED, False),
                ("safe", Color.RED, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM(
            _payload(
                fit={"apple": 0.9, "safe": 0.6},
                danger={"apple": 0.9, "safe": 0.0},
                continue_flag=False,
            )
        )
        guesser_low = LLMGuesser(_scorer(FakeLLM(llm._responses)), lambda_danger=0.0)
        trace_low = guesser_low.guess(view, Clue("clue", 1))
        assert trace_low.guesses[0] == "apple"
        guesser_high = LLMGuesser(_scorer(llm), lambda_danger=2.0)
        trace_high = guesser_high.guess(view, Clue("clue", 1))
        assert trace_high.guesses[0] == "safe"


class TestStopPolicy:
    def test_llm_stop_short_circuits_after_first_guess(self):
        board = _board(
            [
                ("apple", Color.RED, False),
                ("banana", Color.RED, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM(
            _payload(
                fit={"apple": 0.9, "banana": 0.8},
                danger={"apple": 0.0, "banana": 0.0},
                continue_flag=False,
            )
        )
        guesser = LLMGuesser(_scorer(llm))
        trace = guesser.guess(view, Clue("fruit", 3))
        assert trace.guesses == ("apple",)
        assert trace.stop_reason.startswith("llm_gate:llm_stop")

    def test_min_combined_blocks_continue_even_if_llm_says_yes(self):
        # Inspection mode: after committing apple, the next-best score is well
        # below the floor, so the gate stops the loop.
        board = _board(
            [
                ("apple", Color.RED, False),
                ("weak", Color.RED, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM(
            _payload(
                fit={"apple": 0.9, "weak": 0.05},
                danger={"apple": 0.0, "weak": 0.0},
                continue_flag=True,
            )
        )
        guesser = LLMGuesser(
            _scorer(llm),
            gate=ContinueGate(min_combined=0.20),
        )
        trace = guesser.guess(view, Clue("fruit", 2))
        assert trace.guesses == ("apple",)
        assert "min_combined" in trace.stop_reason


class TestPassClue:
    def test_pass_clue_yields_no_guesses(self):
        board = _board([("apple", Color.RED, False)])
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM("{}")
        guesser = LLMGuesser(_scorer(llm))
        trace = guesser.guess(view, Clue(word="", count=0))
        assert trace.guesses == ()
        assert trace.stop_reason == "pass_clue"
        assert llm.calls == []  # no LLM call on pass


class TestEmbeddingFallback:
    def _matrix_for(self, surfaces_and_vecs: list[tuple[str, list[float]]]) -> EmbeddingMatrix:
        surfaces = [s for s, _ in surfaces_and_vecs]
        vectors = np.stack(
            [
                (np.asarray(v, dtype=np.float32) / max(np.linalg.norm(v), 1e-9)).astype(np.float32)
                for _, v in surfaces_and_vecs
            ]
        )
        return EmbeddingMatrix(
            vectors=vectors,
            surfaces=surfaces,
            surface_to_index={s: i for i, s in enumerate(surfaces)},
            provider_id="t",
            vocab_cache_key="t",
        )

    def test_fallback_to_embedding_argmax_on_parse_failure(self):
        matrix = self._matrix_for(
            [
                ("clue", [1.0, 0.0]),
                ("apple", [1.0, 0.0]),  # cosine == 1
                ("car", [-1.0, 0.0]),  # cosine == -1
            ]
        )
        board = _board(
            [
                ("apple", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM("garbage that will not parse")
        guesser = LLMGuesser(
            _scorer(llm),
            embedding_matrix=matrix,
        )
        trace = guesser.guess(view, Clue("clue", 1))
        assert trace.guesses == ("apple",)
        assert trace.llm_steps[0].fallback_path == "embedding_fallback"

    def test_uniform_fallback_when_embedding_coverage_incomplete(self):
        # Matrix lacks 'car' — embedding fallback must skip and roll dice.
        matrix = self._matrix_for(
            [
                ("clue", [1.0, 0.0]),
                ("apple", [1.0, 0.0]),
            ]
        )
        board = _board(
            [
                ("apple", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM("garbage")
        guesser = LLMGuesser(
            _scorer(llm),
            embedding_matrix=matrix,
            rng=np.random.default_rng(0),
        )
        trace = guesser.guess(view, Clue("clue", 1))
        assert len(trace.guesses) == 1
        assert trace.llm_steps[0].fallback_path == "uniform_dead_end"

    def test_uniform_fallback_when_no_matrix_provided(self):
        board = _board(
            [
                ("apple", Color.RED, False),
                ("car", Color.BLUE, False),
            ]
        )
        view = GuesserView(board=board, team=Color.RED)
        llm = FakeLLM("nope")
        guesser = LLMGuesser(_scorer(llm), embedding_matrix=None, rng=np.random.default_rng(1))
        trace = guesser.guess(view, Clue("clue", 1))
        assert trace.llm_steps[0].fallback_path == "uniform_dead_end"


class TestOrchestratorIntegration:
    """Drive the orchestrator with an LLM-primary guesser and assert per-step
    LLM calls happen with refreshed views between reveals."""

    def _full_board(self) -> Board:
        cards = [("r0", Color.RED), ("r1", Color.RED), ("b0", Color.BLUE)]
        cards += [(f"r{i}", Color.RED) for i in range(2, 9)]
        cards += [(f"b{i}", Color.BLUE) for i in range(1, 8)]
        cards += [(f"n{i}", Color.NEUTRAL) for i in range(7)]
        cards += [("ass", Color.ASSASSIN)]
        return Board(
            cards=tuple(Card(word=w, lemma=w, color=c) for w, c in cards),
            first_team=Color.RED,
        )

    def _scripted_spy(self, clue_word: str, n: int):
        from codenames_ai.agent.interfaces import NoLegalClueError, Spymaster
        from codenames_ai.agent.scoring import ScoringWeights
        from codenames_ai.agent.trace import Candidate, ScoreComponents, SpymasterTrace

        class S(Spymaster):
            def __init__(self, c, count):
                self._c = c
                self._count = count
                self._spent = False

            def give_clue(self, view):
                if self._spent:
                    raise NoLegalClueError("scripted spymaster done")
                self._spent = True
                comp = ScoreComponents(expected_reward_raw=0.0)
                cand = Candidate(
                    clue=self._c,
                    targets=(),
                    n=self._count,
                    score=1.0,
                    embedding_score=1.0,
                    components=comp,
                    margin=1.0,
                    zipf=5.0,
                )
                return SpymasterTrace(
                    chosen=cand,
                    top_candidates=(cand,),
                    weights=ScoringWeights.from_risk(0.5),
                    veto_count=0,
                    illegal_count=0,
                )

        return S(clue_word, n)

    def _scripted_guess_responses(self):
        # Three scripted physical-guess decisions — one per reveal we expect.
        # Step 1: r0 wins, continue True.
        # Step 2: r1 wins, continue True.
        # Step 3: r2 wins, continue False.
        all_words = ["r0", "r1", "b0"] + [f"r{i}" for i in range(2, 9)]
        all_words += [f"b{i}" for i in range(1, 8)]
        all_words += [f"n{i}" for i in range(7)] + ["ass"]

        def payload(top: str, continue_flag: bool, exclude: set[str] = set()):
            fit = {w: (0.99 if w == top else 0.05) for w in all_words if w not in exclude}
            danger = {w: 0.0 for w in all_words if w not in exclude}
            return json.dumps({"fit": fit, "danger": danger, "continue": continue_flag})

        return [
            payload("r0", True),
            payload("r1", True, exclude={"r0"}),
            payload("r2", False, exclude={"r0", "r1"}),
        ]

    def test_per_step_llm_calls_with_refreshed_view(self):
        from codenames_ai.game.human import HumanGuesser

        board = self._full_board()
        llm = FakeLLM(self._scripted_guess_responses())
        guesser = LLMGuesser(_scorer(llm), lambda_danger=0.0)

        # BLUE side gets a no-op pass clue so the game ends with RED's turn.
        from codenames_ai.agent.interfaces import NoLegalClueError, Spymaster

        class PassSpy(Spymaster):
            def give_clue(self, view):
                raise NoLegalClueError("blue pass")

        game = Game(
            board,
            red_spymaster=self._scripted_spy("clue", 3),
            red_guesser=guesser,
            blue_spymaster=PassSpy(),
            blue_guesser=HumanGuesser(),
            max_clues=4,
        )
        # Step through one full RED turn.
        game.step()  # spymaster
        game.step()  # guesser

        # Three physical guesses → three LLM calls.
        assert len(llm.calls) == 3
        # Trace has three steps, all attributed to LLM-primary path.
        guess_trace = game.guesser_traces[-1]
        assert isinstance(guess_trace, GuesserTrace)
        assert guess_trace.guesses == ("r0", "r1", "r2")
        assert all(s.fallback_path == "llm_primary" for s in guess_trace.llm_steps)
        # Each user prompt must have shrunk the unrevealed list.
        unrevealed_lines_per_call = [
            sum(1 for line in c["messages"][1]["content"].splitlines() if "[UNREVEALED]" in line)
            for c in llm.calls
        ]
        assert unrevealed_lines_per_call[0] > unrevealed_lines_per_call[1] > unrevealed_lines_per_call[2]

    def test_wrong_color_reveal_ends_turn(self):
        board = self._full_board()
        # First call: pick b0 (BLUE → wrong color, ends turn).
        all_words = [c.word for c in board.cards]
        fit = {w: (0.99 if w == "b0" else 0.05) for w in all_words}
        danger = {w: 0.0 for w in all_words}
        llm = FakeLLM(json.dumps({"fit": fit, "danger": danger, "continue": True}))
        guesser = LLMGuesser(_scorer(llm), lambda_danger=0.0)

        from codenames_ai.agent.interfaces import NoLegalClueError, Spymaster
        from codenames_ai.game.human import HumanGuesser

        class PassSpy(Spymaster):
            def give_clue(self, view):
                raise NoLegalClueError("blue pass")

        game = Game(
            board,
            red_spymaster=self._scripted_spy("clue", 3),
            red_guesser=guesser,
            blue_spymaster=PassSpy(),
            blue_guesser=HumanGuesser(),
            max_clues=4,
        )
        game.step()  # spymaster
        game.step()  # guesser
        # Only one LLM call: wrong-color reveal forces engine to end the turn.
        assert len(llm.calls) == 1
        guess_trace = game.guesser_traces[-1]
        assert guess_trace.guesses == ("b0",)
        assert guess_trace.stop_reason == "turn_ended_by_engine"
