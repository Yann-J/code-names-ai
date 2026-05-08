"""YAML config loading: nested game_zipf, embedding_top_k, legacy keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from codenames_ai.cli.eval_config import EvalAgentConfigFile, load_eval_yaml


def test_nested_clue_zipf(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\n"
        "clue_zipf:\n"
        "  min: 3.5\n"
        "  max: 6.8\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(p)
    assert cfg.clue_zipf.min == pytest.approx(3.5)
    assert cfg.clue_zipf.max == pytest.approx(6.8)


def test_legacy_flat_clue_zipf_keys(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\n"
        "clue_zipf_min: 3.1\n"
        "clue_zipf_max: 6.9\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(p)
    assert cfg.clue_zipf.min == pytest.approx(3.1)
    assert cfg.clue_zipf.max == pytest.approx(6.9)


def test_nested_game_zipf(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\n"
        "game_zipf:\n"
        "  min: 4.2\n"
        "  max: 6.1\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(p)
    assert cfg.game_zipf.min == pytest.approx(4.2)
    assert cfg.game_zipf.max == pytest.approx(6.1)


def test_legacy_flat_game_zipf_keys(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "label: t\n"
        "game_zipf_min: 5.0\n"
        "game_zipf_max: 6.0\n",
        encoding="utf-8",
    )
    cfg, _ = load_eval_yaml(p)
    assert cfg.game_zipf.min == pytest.approx(5.0)
    assert cfg.game_zipf.max == pytest.approx(6.0)


def test_legacy_rerank_top_k_maps_to_embedding_top_k(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("label: t\nrerank_top_k: 42\n", encoding="utf-8")
    cfg, _ = load_eval_yaml(p)
    assert cfg.embedding_top_k == 42


def test_defaults_embedding_top_k_and_trace() -> None:
    cfg = EvalAgentConfigFile()
    assert cfg.embedding_top_k == 200
    assert cfg.top_k_trace == 200
    assert cfg.clue_zipf.min == pytest.approx(3.0)
    assert cfg.clue_zipf.max == pytest.approx(7.0)


def test_clue_allowed_pos_includes_verb_by_default() -> None:
    cfg = EvalAgentConfigFile()
    assert "VERB" in cfg.clue_allowed_pos
