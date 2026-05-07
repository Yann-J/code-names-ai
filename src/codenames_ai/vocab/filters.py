from __future__ import annotations

from pathlib import Path


def is_valid_surface(surface: str, *, min_length: int, allow_hyphens: bool) -> bool:
    """Whether `surface` passes character-set and length checks.

    Always lowercase before calling. Hyphens (when allowed) cannot lead, trail,
    or appear consecutively — those mark malformed tokens.
    """
    if len(surface) < min_length:
        return False
    if not surface:
        return False

    if allow_hyphens:
        if surface.startswith("-") or surface.endswith("-"):
            return False
        if "--" in surface:
            return False
        return all(c.isalpha() or c == "-" for c in surface)

    return surface.isalpha()


def load_exclusions(path: Path | None) -> frozenset[str]:
    """Read a one-word-per-line exclusion list. Lines starting with '#' are comments."""
    if path is None or not path.exists():
        return frozenset()

    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            words.add(word)
    return frozenset(words)
