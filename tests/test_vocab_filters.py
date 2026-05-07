from codenames_ai.vocab.filters import is_valid_surface, load_exclusions


class TestIsValidSurface:
    def test_accepts_simple_lowercase_word(self):
        assert is_valid_surface("apple", min_length=3, allow_hyphens=True)

    def test_rejects_too_short(self):
        assert not is_valid_surface("at", min_length=3, allow_hyphens=True)

    def test_rejects_digits(self):
        assert not is_valid_surface("abc1", min_length=3, allow_hyphens=True)

    def test_rejects_underscores(self):
        assert not is_valid_surface("foo_bar", min_length=3, allow_hyphens=True)

    def test_rejects_apostrophes(self):
        assert not is_valid_surface("don't", min_length=3, allow_hyphens=True)

    def test_accepts_hyphenated_when_allowed(self):
        assert is_valid_surface("well-being", min_length=3, allow_hyphens=True)

    def test_rejects_hyphenated_when_disallowed(self):
        assert not is_valid_surface("well-being", min_length=3, allow_hyphens=False)

    def test_rejects_leading_hyphen(self):
        assert not is_valid_surface("-cat", min_length=3, allow_hyphens=True)

    def test_rejects_trailing_hyphen(self):
        assert not is_valid_surface("cat-", min_length=3, allow_hyphens=True)

    def test_rejects_double_hyphen(self):
        assert not is_valid_surface("foo--bar", min_length=3, allow_hyphens=True)

    def test_rejects_empty_string(self):
        assert not is_valid_surface("", min_length=3, allow_hyphens=True)


class TestLoadExclusions:
    def test_returns_empty_for_none_path(self):
        assert load_exclusions(None) == frozenset()

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert load_exclusions(tmp_path / "missing.txt") == frozenset()

    def test_loads_words_one_per_line(self, tmp_path):
        path = tmp_path / "ex.txt"
        path.write_text("apple\nbanana\ncherry\n")
        assert load_exclusions(path) == frozenset({"apple", "banana", "cherry"})

    def test_strips_whitespace(self, tmp_path):
        path = tmp_path / "ex.txt"
        path.write_text("  apple  \n\tbanana\t\n")
        assert load_exclusions(path) == frozenset({"apple", "banana"})

    def test_lowercases(self, tmp_path):
        path = tmp_path / "ex.txt"
        path.write_text("Apple\nBANANA\n")
        assert load_exclusions(path) == frozenset({"apple", "banana"})

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "ex.txt"
        path.write_text("apple\n\n\nbanana\n")
        assert load_exclusions(path) == frozenset({"apple", "banana"})

    def test_skips_comments(self, tmp_path):
        path = tmp_path / "ex.txt"
        path.write_text("# header\napple\n# inline-style comment\nbanana\n")
        assert load_exclusions(path) == frozenset({"apple", "banana"})
