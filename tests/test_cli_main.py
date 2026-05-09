import logging

import pytest

from codenames_ai.cli.main import (
    build_parser,
    parse_log_level,
    resolve_log_level,
)


def test_parse_log_level_names():
    assert parse_log_level("DEBUG") == logging.DEBUG
    assert parse_log_level("info") == logging.INFO
    assert parse_log_level("  WARNING ") == logging.WARNING


def test_parse_log_level_numeric():
    assert parse_log_level("10") == logging.DEBUG


def test_parse_log_level_empty_or_invalid():
    assert parse_log_level("") is None
    assert parse_log_level("   ") is None
    assert parse_log_level("nope") is None


def test_resolve_log_level_default_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert resolve_log_level(verbose=False) == logging.INFO


def test_resolve_log_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert resolve_log_level(verbose=False) == logging.WARNING


def test_verbose_forces_debug(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert resolve_log_level(verbose=True) == logging.DEBUG


def test_resolve_log_level_invalid_env_warns(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "bogus")
    assert resolve_log_level(verbose=False) == logging.INFO
    assert "invalid LOG_LEVEL" in capsys.readouterr().err


def test_cli_help_exits_zero():
    p = build_parser()
    with pytest.raises(SystemExit) as e:
        p.parse_args(["--help"])
    assert e.value.code == 0


def test_eval_subparser_requires_config():
    p = build_parser()
    with pytest.raises(SystemExit) as e:
        p.parse_args(["eval", "--runs", "2"])
    assert e.value.code != 0
