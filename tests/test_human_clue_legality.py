"""Human spymaster clue checks (board similarity + no reuse)."""

import pandas as pd

from codenames_ai.game.board import generate_board
from codenames_ai.game.models import Clue, Color
from codenames_ai.game.state import TurnEvent
from codenames_ai.vocab.models import Vocabulary, VocabConfig
from codenames_ai.web.game_service import human_clue_validation_error


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


def test_human_clue_allows_pass() -> None:
    assert (
        human_clue_validation_error(
            turn_history=(),
            active_cards=tuple(),
            game_vocab=_vocab(),
            word_lower="",
            count=0,
        )
        is None
    )


def test_human_clue_rejects_empty_word_with_positive_count() -> None:
    err = human_clue_validation_error(
        turn_history=(),
        active_cards=tuple(),
        game_vocab=_vocab(),
        word_lower="",
        count=2,
    )
    assert err is not None


def test_human_clue_rejects_duplicate() -> None:
    hist = (TurnEvent(team=Color.RED, kind="CLUE", clue=Clue(word="ocean", count=1)),)
    err = human_clue_validation_error(
        turn_history=hist,
        active_cards=tuple(),
        game_vocab=_vocab(),
        word_lower="ocean",
        count=1,
    )
    assert err is not None and "already used" in err


def test_human_clue_rejects_surface_on_active_board() -> None:
    board = generate_board(_vocab(), seed=1)
    c0 = board.cards[0]
    err = human_clue_validation_error(
        turn_history=(),
        active_cards=board.active(),
        game_vocab=_vocab(),
        word_lower=c0.word,
        count=1,
    )
    assert err is not None and "board" in err


def test_human_clue_allows_obscure_word_not_near_board() -> None:
    board = generate_board(_vocab(), seed=2)
    clue = "qqqqqqqq"
    assert all(clue not in c.word and c.word not in clue for c in board.cards)
    err = human_clue_validation_error(
        turn_history=(),
        active_cards=board.active(),
        game_vocab=_vocab(),
        word_lower=clue,
        count=1,
    )
    assert err is None
