"""YAML config loading: vocabulary nesting and scoring block validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from codenames_ai.cli.eval_config import EvalAgentConfigFile, load_eval_yaml


def test_nested_clue_zipf(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\n"
        "vocabulary:\n"
        "  clue:\n"
        "    zipf:\n"
        "      min: 3.5\n"
        "      max: 6.8\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(p)
    assert cfg.vocabulary.clue.zipf.min == pytest.approx(3.5)
    assert cfg.vocabulary.clue.zipf.max == pytest.approx(6.8)


def test_nested_game_zipf(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\n"
        "vocabulary:\n"
        "  game:\n"
        "    zipf:\n"
        "      min: 4.2\n"
        "      max: 6.1\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(p)
    assert cfg.vocabulary.game.zipf.min == pytest.approx(4.2)
    assert cfg.vocabulary.game.zipf.max == pytest.approx(6.1)


def test_defaults_embedding_top_k_and_trace() -> None:
    cfg = EvalAgentConfigFile()
    assert cfg.embedding_top_k == 20
    assert cfg.top_k_trace == 200
    assert cfg.vocabulary.clue.zipf.min == pytest.approx(3.0)
    assert cfg.vocabulary.clue.zipf.max == pytest.approx(7.0)


def test_lane_fraction_must_have_7_values(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\nscoring:\n  lane_target_fractions: [0.5, 0.5]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 7 values"):
        load_eval_yaml(p)


def test_lane_fraction_sum_must_be_positive(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\nscoring:\n  lane_target_fractions: [0, 0, 0, 0, 0, 0, 0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to a positive value"):
        load_eval_yaml(p)


def test_clue_allowed_pos_includes_verb_by_default() -> None:
    cfg = EvalAgentConfigFile()
    assert "VERB" in cfg.vocabulary.clue.allowed_pos
