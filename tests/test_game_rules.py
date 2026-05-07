import pytest

from codenames_ai.game.models import Card, Color
from codenames_ai.game.rules import is_legal_clue


def _card(word, lemma=None, color=Color.RED, revealed=False):
    return Card(word=word, lemma=lemma or word, color=color, revealed=revealed)


class TestLemmaStrictness:
    """Default `lemma_substring` strictness is exercised in TestLemmaSubstring;
    here we verify the looser `lemma` mode rejects only exact matches."""

    def test_accepts_unrelated_clue(self):
        assert is_legal_clue(
            clue_surface="banana",
            clue_lemma="banana",
            active_cards=[_card("apple")],
            strictness="lemma",
        )

    def test_rejects_clue_matching_board_surface(self):
        assert not is_legal_clue(
            clue_surface="apple",
            clue_lemma="apple",
            active_cards=[_card("apple")],
            strictness="lemma",
        )

    def test_rejects_clue_lemma_matching_board_lemma(self):
        # clue surface "running" reduces to lemma "run"; board has "ran" lemma "run".
        assert not is_legal_clue(
            clue_surface="running",
            clue_lemma="run",
            active_cards=[_card("ran", lemma="run")],
            strictness="lemma",
        )

    def test_does_not_reject_substring_in_lemma_mode(self):
        # "cat" is a substring of "catnap" but lemma-only mode permits it.
        assert is_legal_clue(
            clue_surface="catnap",
            clue_lemma="catnap",
            active_cards=[_card("cat")],
            strictness="lemma",
        )


class TestLemmaSubstring:
    def test_accepts_unrelated_clue(self):
        assert is_legal_clue(
            clue_surface="banana",
            clue_lemma="banana",
            active_cards=[_card("apple")],
        )

    def test_rejects_compound_with_board_word_as_substring(self):
        assert not is_legal_clue(
            clue_surface="catnap",
            clue_lemma="catnap",
            active_cards=[_card("cat")],
        )

    def test_rejects_clue_when_clue_is_substring_of_board_word(self):
        # Reverse direction: "well" is substring of "wellbeing" board word.
        assert not is_legal_clue(
            clue_surface="well",
            clue_lemma="well",
            active_cards=[_card("wellbeing")],
        )

    def test_rejects_via_clue_lemma_substring(self):
        # clue surface "running" lemma "run" — should fire on board card "rerun".
        assert not is_legal_clue(
            clue_surface="running",
            clue_lemma="run",
            active_cards=[_card("rerun", lemma="rerun")],
        )

    def test_rejects_via_board_lemma_substring(self):
        # board card has lemma "run" (from surface "ran"); clue "outrun" contains it.
        assert not is_legal_clue(
            clue_surface="outrun",
            clue_lemma="outrun",
            active_cards=[_card("ran", lemma="run")],
        )

    def test_revealed_cards_still_considered(self):
        # Active-card filtering happens at the call site (the spymaster passes
        # only `board.active()`); this function trusts its input list.
        assert not is_legal_clue(
            clue_surface="catnap",
            clue_lemma="catnap",
            active_cards=[_card("cat", revealed=False)],
        )

    def test_unknown_strictness_raises(self):
        with pytest.raises(ValueError, match="strictness"):
            is_legal_clue(
                clue_surface="x",
                clue_lemma="x",
                active_cards=[_card("y")],
                strictness="invalid",  # type: ignore[arg-type]
            )

    def test_case_insensitive(self):
        assert not is_legal_clue(
            clue_surface="Apple",
            clue_lemma="Apple",
            active_cards=[_card("apple")],
        )
        assert not is_legal_clue(
            clue_surface="catnap",
            clue_lemma="catnap",
            active_cards=[_card("CAT", lemma="CAT")],
        )

    def test_empty_active_cards_always_legal(self):
        assert is_legal_clue(
            clue_surface="anything",
            clue_lemma="anything",
            active_cards=[],
        )
