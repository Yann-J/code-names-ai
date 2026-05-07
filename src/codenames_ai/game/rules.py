from __future__ import annotations

from typing import Literal

from codenames_ai.game.models import Card

RuleStrictness = Literal["lemma", "lemma_substring"]


def is_legal_clue(
    *,
    clue_surface: str,
    clue_lemma: str,
    active_cards: list[Card] | tuple[Card, ...],
    strictness: RuleStrictness = "lemma_substring",
) -> bool:
    """Whether a single-word clue is legal given the unrevealed board.

    Default strictness ("lemma_substring") rejects:
      - exact surface or lemma match against any active card
      - substring containment in either direction (e.g. "cat" vs "catnap")

    Looser ("lemma") rejects only exact lemma/surface matches.
    """
    if strictness not in ("lemma", "lemma_substring"):
        raise ValueError(f"unknown strictness {strictness!r}")

    cs = clue_surface.lower()
    cl = clue_lemma.lower()

    for card in active_cards:
        bs = card.word.lower()
        bl = card.lemma.lower()
        if cs == bs or cs == bl:
            return False
        if cl == bs or cl == bl:
            return False
        if strictness == "lemma_substring":
            for clue_part in (cs, cl):
                for board_part in (bs, bl):
                    if clue_part == board_part:
                        return False
                    if clue_part in board_part or board_part in clue_part:
                        return False
    return True
