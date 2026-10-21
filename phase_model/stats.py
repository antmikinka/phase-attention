import math


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def paired_tests(a, b):
    """Paired a vs b: exact two-sided sign test + Wilcoxon signed-rank (normal approx, tie-corrected)."""
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    out = {"n": len(a), "mean_diff": sum(x - y for x, y in zip(a, b)) / max(1, len(a)),
           "a_lower_in": sum(1 for x, y in zip(a, b) if x < y), "a_higher_in": sum(1 for x, y in zip(a, b) if x > y)}
    if n == 0:
        out.update(sign_p=1.0, wilcoxon_p=1.0); return out
    k = sum(1 for x in d if x < 0)
    out["sign_p"] = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)
    order = sorted(range(n), key=lambda i: abs(d[i])); r = [0.0] * n; i = 0; ties = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for q in range(i, j + 1):
            r[order[q]] = (i + j) / 2 + 1
        t = j - i + 1; ties += t ** 3 - t; i = j + 1
    wp = sum(r[i] for i in range(n) if d[i] > 0); mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - ties / 48)
    out["wilcoxon_p"] = min(1.0, 2 * _norm_sf((abs(wp - mu) - 0.5) / sd)) if sd > 0 else 1.0
    return out
