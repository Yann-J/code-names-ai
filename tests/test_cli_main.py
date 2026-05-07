import pytest

from codenames_ai.cli.main import build_parser


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
