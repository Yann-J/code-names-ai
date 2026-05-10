"""Pure-function tests for the LLM-primary guesser policy engine."""

from __future__ import annotations

import pytest

from codenames_ai.agent.llm_guess_policy import (
    ContinueGate,
    argmax_combined,
    combined_scores,
    evaluate_continue_gate,
    margin_to_second,
)


class TestCombinedScores:
    def test_simple_combine(self):
        fit = {"apple": 0.9, "car": 0.4}
        danger = {"apple": 0.1, "car": 0.6}
        out = combined_scores(fit, danger, lambda_danger=0.5)
        assert out["apple"] == pytest.approx(0.9 - 0.05)
        assert out["car"] == pytest.approx(0.4 - 0.30)

    def test_zero_lambda_keeps_fit(self):
        fit = {"a": 0.7, "b": 0.3}
        danger = {"a": 1.0, "b": 1.0}
        out = combined_scores(fit, danger, lambda_danger=0.0)
        assert out == {"a": 0.7, "b": 0.3}

    def test_drops_words_missing_from_danger(self):
        fit = {"a": 0.7, "b": 0.3}
        danger = {"a": 0.1}
        out = combined_scores(fit, danger, lambda_danger=0.5)
        assert "b" not in out
        assert out["a"] == pytest.approx(0.7 - 0.05)


class TestArgmaxCombined:
    def test_picks_highest(self):
        scores = {"a": 0.1, "b": 0.9, "c": 0.5}
        assert argmax_combined(scores) == "b"

    def test_tie_break_uses_candidate_order(self):
        scores = {"x": 0.5, "y": 0.5}
        assert argmax_combined(scores, candidates=("y", "x")) == "y"
        assert argmax_combined(scores, candidates=("x", "y")) == "x"

    def test_lexicographic_tie_break_when_no_candidates(self):
        scores = {"banana": 0.5, "apple": 0.5}
        assert argmax_combined(scores) == "apple"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            argmax_combined({})

    def test_candidates_filter_to_scored_subset(self):
        scores = {"a": 0.9, "b": 0.5}
        # 'c' is in candidates but not scored — argmax must skip it.
        assert argmax_combined(scores, candidates=("c", "a", "b")) == "a"

    def test_candidates_with_no_overlap_raises(self):
        scores = {"a": 0.9}
        with pytest.raises(ValueError):
            argmax_combined(scores, candidates=("x",))


class TestMarginToSecond:
    def test_positive_margin(self):
        scores = {"a": 0.9, "b": 0.4, "c": 0.1}
        assert margin_to_second(scores, "a") == pytest.approx(0.5)

    def test_zero_when_only_one_card(self):
        assert margin_to_second({"only": 0.5}, "only") == 0.0

    def test_zero_when_chosen_missing(self):
        assert margin_to_second({"a": 0.9, "b": 0.4}, "ghost") == 0.0


class TestEvaluateContinueGate:
    def test_llm_stop_short_circuits(self):
        decision = evaluate_continue_gate(
            llm_continue=False,
            chosen="a",
            next_combined={"b": 0.8},
            next_fit={"b": 0.9},
            gate=ContinueGate(),
            attempts_remaining_after=5,
        )
        assert decision.proceed is False
        assert decision.reason == "llm_stop"

    def test_attempts_exhausted_overrides_continue(self):
        decision = evaluate_continue_gate(
            llm_continue=True,
            chosen="a",
            next_combined={"b": 0.8},
            next_fit={"b": 0.9},
            gate=ContinueGate(),
            attempts_remaining_after=0,
        )
        assert decision.proceed is False
        assert decision.reason == "attempts_exhausted"

    def test_min_combined_blocks_low_confidence(self):
        decision = evaluate_continue_gate(
            llm_continue=True,
            chosen="a",
            next_combined={"b": 0.05},
            next_fit={"b": 0.9},
            gate=ContinueGate(min_combined=0.20),
            attempts_remaining_after=None,
        )
        assert decision.proceed is False
        assert decision.reason == "min_combined"

    def test_min_margin_blocks_close_runners(self):
        decision = evaluate_continue_gate(
            llm_continue=True,
            chosen="a",
            next_combined={"b": 0.5, "c": 0.49},
            next_fit={"b": 0.6, "c": 0.6},
            gate=ContinueGate(min_margin_to_second=0.10),
            attempts_remaining_after=None,
        )
        assert decision.proceed is False
        assert decision.reason == "min_margin"

    def test_min_fit_blocks_assassin_shaped(self):
        decision = evaluate_continue_gate(
            llm_continue=True,
            chosen="a",
            next_combined={"b": 0.4},
            next_fit={"b": 0.05},
            gate=ContinueGate(min_fit=0.20),
            attempts_remaining_after=None,
        )
        assert decision.proceed is False
        assert decision.reason == "min_fit"

    def test_no_more_unrevealed(self):
        decision = evaluate_continue_gate(
            llm_continue=True,
            chosen="a",
            next_combined={},
            next_fit={},
            gate=ContinueGate(),
            attempts_remaining_after=None,
        )
        assert decision.proceed is False
        assert decision.reason == "no_more_unrevealed"

    def test_passes_when_all_floors_clear(self):
        decision = evaluate_continue_gate(
            llm_continue=True,
            chosen="a",
            next_combined={"b": 0.7, "c": 0.2},
            next_fit={"b": 0.8, "c": 0.3},
            gate=ContinueGate(min_combined=0.5, min_margin_to_second=0.3, min_fit=0.5),
            attempts_remaining_after=3,
        )
        assert decision.proceed is True
        assert decision.reason == "gate_passed"
