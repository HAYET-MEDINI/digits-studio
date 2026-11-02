"""Conversion d'un entier en toutes lettres (français, orthographe rectifiée classique)."""
U = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
     "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize"]
T = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}


def _sub100(n, fin):
    if n < 17:
        return U[n]
    if n < 20:
        return "dix-" + U[n - 10]
    if n < 70:
        t, u = divmod(n, 10)
        if u == 0:
            return T[t]
        return T[t] + (" et un" if u == 1 else "-" + U[u])
    if n < 80:
        r = n - 60
        return "soixante et onze" if r == 11 else "soixante-" + _sub100(r, fin)
    r = n - 80
    if r == 0:
        return "quatre-vingts" if fin else "quatre-vingt"
    return "quatre-vingt-" + _sub100(r, fin)


def _grp(n, fin):
    h, r = divmod(n, 100)
    out = []
    if h:
        out.append(("cent" if h == 1 else U[h] + " cent") + ("s" if h > 1 and r == 0 and fin else ""))
    if r:
        out.append(_sub100(r, fin))
    return " ".join(out)


def to_words(n):
    if n == 0:
        return "zéro"
    if n >= 10 ** 12:
        return ""
    parts = []
    for scale, name in ((10 ** 9, "milliard"), (10 ** 6, "million")):
        c, n = divmod(n, scale)
        if c:
            parts.append(_grp(c, True) + " " + name + ("s" if c > 1 else ""))
    c, n = divmod(n, 1000)
    if c:
        parts.append("mille" if c == 1 else _grp(c, False) + " mille")
    if n:
        parts.append(_grp(n, True))
    return " ".join(parts)
