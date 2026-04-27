"""Qualitative inspection helper: print N random pairs from each cell, plus
extra style metrics (critique-specific markers, opening-token diversity)."""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def critique_specific(a):
    pats = {
        "more X (decisive/practical/...)": r"\bmore (decisive|practical|actionable|usable|empowering|motivating|compelling|useful|insightful|effective|concrete)\b",
        "A is stronger because": r"\b(answer|response|reply|argument) [ab12c] is (the )?(better|stronger|more|most)\b",
        "this answer is [adj]": r"\bthis (answer|advice|response|reply|statement|argument) is (excellent|effective|sharp|practical|usable|useful|compelling|good|solid|clear|sound|correct|fundamentally correct)\b",
        "cuts through": r"\bcuts through\b",
        "(would|could) be stronger": r"\b(would|could) be stronger\b",
        "strongest part of": r"\b(strongest|best) part of (this|the)\b",
        "stronger version / better framing / revised version": r"\b(a (stronger|better) version|a better framing|revised version|stronger version|better framing)\b",
        "this advice is": r"\bthis (advice|statement|answer|response) is\b",
    }
    n = len(a)
    out = {}
    for lbl, pat in pats.items():
        c = sum(1 for x in a if re.search(pat, x.lower()))
        out[lbl] = round(100 * c / n, 1)
    return out


def opening_diversity(a):
    """Top-5 opening-2-word frequencies (low diversity = high template)."""
    op = Counter(" ".join(x.split()[:2]).lower() for x in a)
    return op.most_common(5)


def main():
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    paths = sys.argv[2:]
    if not paths:
        print(__doc__)
        sys.exit(1)
    rng = random.Random(0)
    for p in paths:
        rs = load(p)
        a = [r["messages"][1]["content"] for r in rs]
        u = [r["messages"][0]["content"] for r in rs]
        print(f"\n{'='*80}\n  {Path(p).name} (n={len(rs)})\n{'='*80}")
        cs = critique_specific(a)
        print("Critique-specific markers:")
        for k, v in cs.items():
            print(f"  {k:<55} {v:>5.1f}%")
        print("Top opening 2-words:")
        for op, c in opening_diversity(a):
            print(f"  '{op}' {c}")
        idxs = rng.sample(range(len(rs)), min(n_show, len(rs)))
        print(f"\n{n_show} random samples:")
        for i in idxs:
            print(f"  --- U: {u[i][:300]}")
            print(f"      A: {a[i]}")


if __name__ == "__main__":
    main()
