import math

import pytest

from codenames_ai.agent.scoring import ScoringWeights, freq_bonus


class TestScoringWeightsFromRisk:
    def test_clamped_to_zero(self):
        # risk=0 should clamp negative inputs.
        cautious = ScoringWeights.from_risk(-0.5)
        ref = ScoringWeights.from_risk(0.0)
        assert cautious == ref

    def test_clamped_to_one(self):
        aggressive = ScoringWeights.from_risk(2.0)
        ref = ScoringWeights.from_risk(1.0)
        assert aggressive == ref

    def test_cautious_demands_more_margin_than_aggressive(self):
        cautious = ScoringWeights.from_risk(0.0)
        aggressive = ScoringWeights.from_risk(1.0)
        assert cautious.margin_floor > aggressive.margin_floor
        assert cautious.margin_weight > aggressive.margin_weight

    def test_aggressive_rewards_ambition_more(self):
        cautious = ScoringWeights.from_risk(0.0)
        aggressive = ScoringWeights.from_risk(1.0)
        assert aggressive.ambition_weight > cautious.ambition_weight

    def test_cautious_penalizes_assassin_more(self):
        cautious = ScoringWeights.from_risk(0.0)
        aggressive = ScoringWeights.from_risk(1.0)
        assert cautious.assassin_weight > aggressive.assassin_weight
        assert cautious.assassin_ceiling < aggressive.assassin_ceiling

    def test_intermediate_risk_lies_between_extremes(self):
        cautious = ScoringWeights.from_risk(0.0)
        mid = ScoringWeights.from_risk(0.5)
        aggressive = ScoringWeights.from_risk(1.0)
        # margin_floor monotonically decreases with risk
        assert cautious.margin_floor >= mid.margin_floor >= aggressive.margin_floor
        # ambition_weight monotonically increases with risk
        assert cautious.ambition_weight <= mid.ambition_weight <= aggressive.ambition_weight


class TestFreqBonus:
    def test_zero_at_zipf_3(self):
        # tanh((3 - 3) / 2) == 0
        assert freq_bonus(3.0, freq_weight=0.10) == pytest.approx(0.0)

    def test_positive_above_pivot(self):
        assert freq_bonus(5.0, freq_weight=0.10) > 0

    def test_negative_below_pivot(self):
        assert freq_bonus(2.0, freq_weight=0.10) < 0

    def test_saturates_at_high_zipf(self):
        # tanh saturates near 1; at Zipf 8 the bonus should be ~ freq_weight * 0.999
        assert freq_bonus(8.0, freq_weight=0.10) == pytest.approx(
            0.10 * math.tanh(2.5), abs=1e-6
        )
        # Increasing Zipf further barely moves the value.
        assert freq_bonus(9.0, freq_weight=0.10) - freq_bonus(8.0, freq_weight=0.10) < 0.01

    def test_scales_with_freq_weight(self):
        a = freq_bonus(5.0, freq_weight=0.10)
        b = freq_bonus(5.0, freq_weight=0.20)
        assert b == pytest.approx(2 * a)
