from digits_studio.numbers_fr import to_words

CASES = {
    0: "zéro", 21: "vingt et un", 71: "soixante et onze", 80: "quatre-vingts",
    81: "quatre-vingt-un", 91: "quatre-vingt-onze", 100: "cent", 200: "deux cents",
    280: "deux cent quatre-vingts", 1000: "mille", 1980: "mille neuf cent quatre-vingts",
    2026: "deux mille vingt-six", 80000: "quatre-vingt mille", 172000: "cent soixante-douze mille",
    1_000_000: "un million", 2_000_000: "deux millions", 4052: "quatre mille cinquante-deux",
}


def test_known_numbers():
    for n, words in CASES.items():
        assert to_words(n) == words, f"{n} -> {to_words(n)!r}, attendu {words!r}"


def test_out_of_range_returns_empty():
    assert to_words(10 ** 12) == ""
