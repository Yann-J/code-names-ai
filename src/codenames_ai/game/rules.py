from __future__ import annotations

from typing import Literal

from codenames_ai.game.models import Card

RuleStrictness = Literal["lemma", "lemma_substring"]

# Reject clue/board pairs that differ by at most one edit when both sides are long
# enough that single-letter typos and US/UK spelling variants matter (e.g. rumors/rumours).
_MIN_LEN_LEVENSHTEIN_1 = 5
_MIN_DERIV_ROOT = 4
_MIN_DERIV_COMMON_PREFIX = 5
_DERIVATIONAL_SUFFIXES = (
    "ication",
    "ation",
    "ition",
    "ative",
    "izing",
    "ising",
    "ingly",
    "ably",
    "ator",
    "izer",
    "iser",
    "able",
    "ible",
    "ance",
    "ence",
    "ment",
    "ness",
    "tion",
    "sion",
    "ally",
    "edly",
    "ical",
    "ates",
    "ated",
    "ating",
    "ions",
    "ists",
    "isms",
    "ive",
    "ous",
    "ing",
    "ers",
    "ors",
    "ies",
    "ity",
    "ism",
    "ist",
    "ate",
    "ion",
    "al",
    "ic",
    "er",
    "or",
    "ed",
    "es",
    "s",
)


def _levenshtein_at_most_one(a: str, b: str) -> bool:
    """True if Levenshtein distance between a and b is 0 or 1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Ensure s1 is the shorter string (allows one insert into s1 to get s2).
    s1, s2 = (a, b) if la <= lb else (b, a)
    i = j = 0
    used = False
    while i < len(s1) and j < len(s2):
        if s1[i] == s2[j]:
            i += 1
            j += 1
        elif used:
            return False
        else:
            used = True
            if len(s1) == len(s2):
                i += 1
                j += 1
            else:
                j += 1
    return True


def _spelling_too_close(clue_part: str, board_part: str) -> bool:
    if len(clue_part) < _MIN_LEN_LEVENSHTEIN_1 or len(board_part) < _MIN_LEN_LEVENSHTEIN_1:
        return False
    return _levenshtein_at_most_one(clue_part, board_part)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _derivational_root(word: str) -> str | None:
    for suffix in _DERIVATIONAL_SUFFIXES:
        if not word.endswith(suffix):
            continue
        root = word[: -len(suffix)]
        if len(root) >= _MIN_DERIV_ROOT:
            return root
    return None


def _derivationally_related(a: str, b: str) -> bool:
    ar = _derivational_root(a)
    br = _derivational_root(b)
    if ar is None or br is None:
        return False
    return _common_prefix_len(ar, br) >= _MIN_DERIV_COMMON_PREFIX


def is_legal_clue(
    *,
    clue_surface: str,
    clue_lemma: str,
    active_cards: list[Card] | tuple[Card, ...],
    strictness: RuleStrictness = "lemma_substring",
) -> bool:
    """Whether a single-word clue is legal given the unrevealed board.

    Always rejects when clue surface or lemma is within Levenshtein distance 1 of a
    board surface or lemma, if both strings are at least 5 letters (e.g. rumors/rumours).

    Default strictness ("lemma_substring") additionally rejects:
      - exact surface or lemma match against any active card
      - substring containment in either direction (e.g. "cat" vs "catnap")

    Looser ("lemma") rejects exact lemma/surface matches and the spelling-distance rule
    above, but not substring containment.
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
        for clue_part in (cs, cl):
            for board_part in (bs, bl):
                if _spelling_too_close(clue_part, board_part):
                    return False
                if _derivationally_related(clue_part, board_part):
                    return False
        if strictness == "lemma_substring":
            for clue_part in (cs, cl):
                for board_part in (bs, bl):
                    if clue_part == board_part:
                        return False
                    if clue_part in board_part or board_part in clue_part:
                        return False
    return True
