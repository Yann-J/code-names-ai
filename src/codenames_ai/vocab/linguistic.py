from __future__ import annotations

from typing import Protocol


class LinguisticProcessor(Protocol):
    """POS-tag and lemmatize a list of single-word inputs.

    Returns one `(lemma, pos)` per input, in the same order.
    """

    def analyze_batch(self, words: list[str]) -> list[tuple[str, str]]:
        ...


_SPACY_MODEL_FOR: dict[str, str] = {
    "en": "en_core_web_sm",
}


class SpacyLinguisticProcessor:
    """`LinguisticProcessor` backed by spaCy.

    spaCy's POS tagger expects sentence context; we run it on bare single-word
    (or hyphenated) inputs and accept that some POS tags reflect the most-likely
    standalone reading rather than a specific contextual usage. For Code Names
    purposes that's the right behavior — we're picking words that *can* function
    as nouns/adjectives, not annotating specific usages.
    """

    def __init__(self, nlp) -> None:
        self._nlp = nlp

    @classmethod
    def for_language(cls, language: str) -> SpacyLinguisticProcessor:
        try:
            import spacy
        except ImportError as e:
            raise RuntimeError(
                "spaCy is required to build a vocabulary. Install with: pip install spacy"
            ) from e

        model_name = _SPACY_MODEL_FOR.get(language)
        if model_name is None:
            raise ValueError(
                f"No spaCy model configured for language {language!r}. "
                f"Add an entry to vocab.linguistic._SPACY_MODEL_FOR."
            )
        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
        except OSError as e:
            raise RuntimeError(
                f"spaCy model {model_name!r} is not installed.\n"
                f"Run: python -m spacy download {model_name}"
            ) from e
        return cls(nlp)

    def analyze_batch(self, words: list[str]) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for surface, doc in zip(words, self._nlp.pipe(words, batch_size=500), strict=True):
            head = _pick_head_token(doc)
            if head is None:
                # Pathological input; treat as unknown so the POS filter drops it.
                results.append((surface, "X"))
                continue
            # For hyphenated inputs, spaCy splits on hyphens — preserve the surface
            # form as the lemma so the artifact retains the original token.
            lemma = surface if "-" in surface else head.lemma_.lower()
            results.append((lemma, head.pos_))
        return results


def _pick_head_token(doc):
    """Pick the most informative token from a spaCy `Doc` for a single-word input."""
    for tok in doc:
        if tok.pos_ not in ("PUNCT", "SPACE", "SYM"):
            return tok
    return doc[0] if len(doc) > 0 else None
