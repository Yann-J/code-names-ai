"""Dynamic risk maths and modulation (Github issue #1)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from codenames_ai.agent.risk_context import (
    DynamicRiskPolicy,
    apply_effective_risk,
    compute_effective_risk,
    modulate_scoring_weights,
    modulate_stop_policy,
    objectives_delta,
    unrevealed_objective_count,
)
from codenames_ai.agent.scoring import ScoringWeights, StopPolicy
from codenames_ai.game.models import Board, Card, Color


def _filled_board(team_cards: tuple[Card, ...]) -> Board:
    assassin = Card(word="__assassin__", lemma="__assassin__", color=Color.ASSASSIN)
    n = len(team_cards)
    filler = tuple(
        Card(word=f"n{i}", lemma=f"n{i}", color=Color.NEUTRAL) for i in range(25 - n - 1)
    )
    return Board(cards=team_cards + filler + (assassin,), first_team=Color.RED)


def _board_progress(
    *,
    red_remaining: int,
    blue_remaining: int,
    red_total: int = 9,
    blue_total: int = 8,
) -> Board:
    assert 0 <= red_remaining <= red_total
    assert 0 <= blue_remaining <= blue_total
    cards: list[Card] = []
    for i in range(red_total):
        cards.append(
            Card(
                word=f"r{i}",
                lemma=f"r{i}",
                color=Color.RED,
                revealed=i < (red_total - red_remaining),
            )
        )
    for i in range(blue_total):
        cards.append(
            Card(
                word=f"b{i}",
                lemma=f"b{i}",
                color=Color.BLUE,
                revealed=i < (blue_total - blue_remaining),
            )
        )
    return _filled_board(tuple(cards))


def test_behind_team_gets_higher_effective_risk_than_ahead_when_enabled() -> None:
    ahead = _board_progress(red_remaining=6, blue_remaining=8)
    behind = _board_progress(red_remaining=8, blue_remaining=6)
    pol = DynamicRiskPolicy(enabled=True, s=0.12, min_risk=0.0, max_risk=1.0)
    base = 0.5
    d_ahead = objectives_delta(ahead, Color.RED)[2]
    d_behind = objectives_delta(behind, Color.RED)[2]
    assert d_behind < d_ahead, "fixture ordering"
    e_ahead = compute_effective_risk(base, d_ahead, pol)
    e_behind = compute_effective_risk(base, d_behind, pol)
    assert e_behind > e_ahead


def test_effective_risk_clamped_to_min_max_under_blowouts() -> None:
    far_behind = _board_progress(red_remaining=9, blue_remaining=2)
    far_ahead = _board_progress(red_remaining=3, blue_remaining=8)
    pol = DynamicRiskPolicy(enabled=True, s=5.0, min_risk=0.2, max_risk=0.8)
    base = 0.5
    e_low = compute_effective_risk(base, objectives_delta(far_ahead, Color.RED)[2], pol)
    e_high = compute_effective_risk(base, objectives_delta(far_behind, Color.RED)[2], pol)
    assert e_low >= pol.min_risk - 1e-9 and e_low <= pol.max_risk + 1e-9
    assert e_high >= pol.min_risk - 1e-9 and e_high <= pol.max_risk + 1e-9


def test_delta_definition_symmetric_under_team_swap_red_blue() -> None:
    """Same geometric progress, swapping viewpoint flips Δ sign magnitude."""

    mid = _board_progress(red_remaining=5, blue_remaining=6)
    r_ours = unrevealed_objective_count(mid, Color.RED)
    r_theirs = unrevealed_objective_count(mid, Color.BLUE)
    b_ours = unrevealed_objective_count(mid, Color.BLUE)
    b_theirs = unrevealed_objective_count(mid, Color.RED)
    d_red = objectives_delta(mid, Color.RED)[2]
    d_blue = objectives_delta(mid, Color.BLUE)[2]
    assert r_ours == b_theirs and r_theirs == b_ours
    assert d_red == pytest.approx(-d_blue)


def test_delta_zero_keeps_baselines_under_modulation() -> None:
    board = _board_progress(red_remaining=6, blue_remaining=6)
    pol = DynamicRiskPolicy(enabled=True, s=1.0, beta_margin_floor=0.9, beta_assassin_ceiling=0.9)
    ours, theirs, delta = objectives_delta(board, Color.RED)
    assert delta == 0.0
    assert ours == theirs
    base_r = 0.55
    eff = compute_effective_risk(base_r, delta, pol)
    assert eff == pytest.approx(base_r)
    w0 = replace(ScoringWeights.from_risk(base_r), margin_floor=0.07)
    w1 = modulate_scoring_weights(
        w0, base_risk=base_r, effective_risk=eff, policy=pol
    )
    assert w1.margin_floor == pytest.approx(w0.margin_floor)
    assert w1.assassin_ceiling == pytest.approx(w0.assassin_ceiling)
    s0 = StopPolicy.from_risk(base_r)
    s1 = modulate_stop_policy(s0, base_risk=base_r, effective_risk=eff, policy=pol)
    assert s1.confidence_floor == pytest.approx(s0.confidence_floor)
    assert s1.bonus_gap_threshold == pytest.approx(s0.bonus_gap_threshold)


def test_dynamic_disabled_leaves_vectors_unchanged() -> None:
    pol = DynamicRiskPolicy(enabled=False)
    w = ScoringWeights.from_risk(0.72)
    s = StopPolicy.from_risk(0.72)
    w2, s2 = apply_effective_risk(
        w,
        s,
        base_risk=0.72,
        effective_risk=0.99,
        policy=pol,
    )
    assert w2 == w and s2 == s

