from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class FrequencyProvider(ABC):
    """Yields `(surface, zipf)` pairs from a frequency source.

    Implementations must yield in **descending Zipf order** so callers can break
    once they fall below `zipf_min`.
    """

    @abstractmethod
    def iter_range(
        self,
        *,
        language: str,
        zipf_min: float,
        zipf_max: float,
    ) -> Iterator[tuple[str, float]]:
        ...


class WordfreqProvider(FrequencyProvider):
    """`FrequencyProvider` backed by the `wordfreq` library."""

    def iter_range(
        self,
        *,
        language: str,
        zipf_min: float,
        zipf_max: float,
    ) -> Iterator[tuple[str, float]]:
        from wordfreq import iter_wordlist, zipf_frequency

        for surface in iter_wordlist(language):
            z = zipf_frequency(surface, language)
            if z < zipf_min:
                # iter_wordlist is sorted by frequency desc → safe to break.
                return
            if z > zipf_max:
                continue
            yield surface, z
