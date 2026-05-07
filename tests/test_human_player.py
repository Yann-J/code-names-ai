import pandas as pd
import pytest

from codenames_ai.game.board import generate_board
from codenames_ai.game.human import (
    HumanGuesser,
    HumanSpymaster,
    trivial_guesser_trace,
    trivial_spymaster_trace,
)
from codenames_ai.game.models import Clue, Color, GuesserView, SpymasterView
from codenames_ai.vocab.models import Vocabulary, VocabConfig


def _vocab() -> Vocabulary:
    config = VocabConfig(
        language="en",
        zipf_min=4.0,
        zipf_max=6.5,
        allowed_pos=frozenset({"NOUN"}),
    )
    df = pd.DataFrame(
        [
            {
                "surface": f"w{i:03d}",
                "lemma": f"w{i:03d}",
                "zipf": 5.0,
                "pos": "NOUN",
            }
            for i in range(40)
        ]
    )
    return Vocabulary(config=config, df=df)


@pytest.fixture
def board():
    return generate_board(_vocab(), seed=0)


def test_human_spymaster_prepare_and_consume(board):
    h = HumanSpymaster()
    tr = trivial_spymaster_trace("ocean", targets=(), n=2)
    h.prepare(tr)
    out = h.give_clue(SpymasterView(board=board, team=Color.RED))
    assert out is tr


def test_human_spymaster_double_prepare_errors():
    h = HumanSpymaster()
    h.prepare(trivial_spymaster_trace("x", targets=(), n=1))
    with pytest.raises(RuntimeError, match="prepare"):
        h.prepare(trivial_spymaster_trace("y", targets=(), n=1))


def test_human_guesser_prepare_and_consume(board):
    h = HumanGuesser()
    tr = trivial_guesser_trace(("a", "b"))
    h.prepare(tr)
    out = h.guess(
        GuesserView(board=board, team=Color.RED),
        Clue(word="ocean", count=2),
    )
    assert out is tr
