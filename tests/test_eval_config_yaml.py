"""YAML config loading: vocabulary nesting and scoring block validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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
    assert cfg.scoring.embedding_top_k == 20
    assert cfg.top_k_trace == 200
    assert cfg.guesser.sampling_temperature == pytest.approx(0.0)
    assert cfg.guesser.sampling_top_k == 0
    assert cfg.risk.base_risk == pytest.approx(0.5)
    assert cfg.dynamic_risk.enabled is False
    assert cfg.dynamic_risk.min_risk <= cfg.dynamic_risk.max_risk
    assert cfg.vocabulary.clue.zipf.min == pytest.approx(3.0)
    assert cfg.vocabulary.clue.zipf.max == pytest.approx(7.0)


def test_unknown_dynamic_risk_field_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\ndynamic_risk:\n  enabled: false\n  surprise_key: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_eval_yaml(p)


def test_dynamic_risk_min_gt_max_validation(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\ndynamic_risk:\n  min_risk: 0.9\n  max_risk: 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_eval_yaml(p)


def test_unknown_scoring_field_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\nscoring:\n  legacy_lane_target_fractions: [1]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_eval_yaml(p)


def test_lane_max_n_valid(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("label: t\nscoring:\n  lane_max_n: 5\n", encoding="utf-8")
    cfg, _ = load_eval_yaml(p)
    assert cfg.scoring.lane_max_n == 5


def test_clue_allowed_pos_includes_verb_by_default() -> None:
    cfg = EvalAgentConfigFile()
    assert "VERB" in cfg.vocabulary.clue.allowed_pos


def test_legacy_flat_risk_scalar_in_yaml(tmp_path: Path) -> None:
    """Top-level ``risk: 0.42`` is accepted as ``risk.base_risk`` for older configs."""
    p = tmp_path / "cfg.yaml"
    p.write_text("label: t\nrisk: 0.42\n", encoding="utf-8")
    cfg, _ = load_eval_yaml(p)
    assert cfg.risk.base_risk == pytest.approx(0.42)


def test_extends_merges_parent_and_child_wins(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "label: parent\nrisk:\n  base_risk: 0.3\nvocabulary:\n  language: en\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: parent.yaml\nlabel: child\nrisk:\n  base_risk: 0.9\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(child)
    assert cfg.label == "child"
    assert cfg.risk.base_risk == pytest.approx(0.9)
    assert cfg.vocabulary.language == "en"
