from __future__ import annotations

import random

from codenames_ai.game.models import Board, Card, Color
from codenames_ai.vocab.models import Vocabulary


def generate_board(
    vocab: Vocabulary,
    *,
    seed: int,
    first_team: Color | None = None,
) -> Board:
    """Sample a deterministic 25-card board from `vocab`.

    Color assignment: 9 cards for `first_team`, 8 for the other team, 7 neutral,
    1 assassin. If `first_team` is None it is chosen by the same rng so the
    whole board is determined by the seed.
    """
    if len(vocab) < 25:
        raise ValueError(f"vocab has only {len(vocab)} entries; need at least 25")

    rng = random.Random(seed)
    if first_team is None:
        first_team = rng.choice([Color.RED, Color.BLUE])
    if not first_team.is_team:
        raise ValueError(f"first_team must be RED or BLUE, got {first_team!r}")

    indices = rng.sample(range(len(vocab)), 25)
    df = vocab.df
    surfaces = df["surface"].iloc[indices].tolist()
    lemmas = df["lemma"].iloc[indices].tolist()

    other_team = first_team.opponent()
    colors = (
        [first_team] * 9
        + [other_team] * 8
        + [Color.NEUTRAL] * 7
        + [Color.ASSASSIN] * 1
    )
    rng.shuffle(colors)

    cards = tuple(
        Card(word=str(s), lemma=str(l), color=c)
        for s, l, c in zip(surfaces, lemmas, colors, strict=True)
    )
    return Board(cards=cards, first_team=first_team)
