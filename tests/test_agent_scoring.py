import pytest

from codenames_ai.agent.scoring import ScoringWeights


class TestScoringWeightsFromRisk:
    def test_clamped_to_zero(self):
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
        assert cautious.assassin_ceiling < aggressive.assassin_ceiling

    def test_cautious_mc_sharper_than_aggressive(self):
        cautious = ScoringWeights.from_risk(0.0)
        aggressive = ScoringWeights.from_risk(1.0)
        assert cautious.mc_temperature < aggressive.mc_temperature

    def test_mc_rank_bias_larger_when_cautious(self):
        """Rank bias concentrates softmax on top cosine matches when gaps are shallow."""
        cautious = ScoringWeights.from_risk(0.0)
        aggressive = ScoringWeights.from_risk(1.0)
        assert cautious.mc_rank_bias > aggressive.mc_rank_bias

    def test_intermediate_mc_rank_bias_between_extremes(self):
        cautious = ScoringWeights.from_risk(0.0)
        mid = ScoringWeights.from_risk(0.5)
        aggressive = ScoringWeights.from_risk(1.0)
        assert cautious.mc_rank_bias >= mid.mc_rank_bias >= aggressive.mc_rank_bias

    def test_intermediate_risk_lies_between_extremes(self):
        cautious = ScoringWeights.from_risk(0.0)
        mid = ScoringWeights.from_risk(0.5)
        aggressive = ScoringWeights.from_risk(1.0)
        assert cautious.margin_floor >= mid.margin_floor >= aggressive.margin_floor
        assert cautious.mc_temperature <= mid.mc_temperature <= aggressive.mc_temperature

    def test_default_rewards_stable(self):
        w = ScoringWeights.from_risk(0.5)
        assert w.reward_friendly == pytest.approx(1.0)
        assert w.lane_max_n == 7
        assert w.mc_trials == 96
