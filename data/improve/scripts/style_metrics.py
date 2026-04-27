"""Compute style metrics for any *_strong.jsonl with messages-format rows.

Metrics:
  n           — row count
  avg_w       — mean assistant word count
  uniq50%     — % rows with unique first-50-char prefix
  uniq8w%     — % rows with unique first-8-words
  valid%      — sum of validation/affirmation marker hits (rough sycophancy %)
  hedge%      — sum of hedge / referral / "but/however" hit rates

Usage:
  python -m data.improve.scripts.style_metrics path1.jsonl path2.jsonl ...
"""
import json
import re
import statistics
import sys
from pathlib import Path

VALID = [
    r"\bfantastic\b", r"\bexcellent\b",
    r"\bgreat (idea|approach|way|strategy|question|point)\b",
    r"\b(very )?(practical|reasonable|smart|wise|sensible) (idea|approach|move|call|judgment)\b",
    r"\b(you'?re|you are) (in the best position|right to|wise to|absolutely right)\b",
    r"\bthat'?s a (very|great|wise|smart|fantastic|wonderful|practical|compassionate|reasonable|sensible)\b",
    r"\bperfectly (safe|fine|reasonable|natural|normal)\b",
    r"\bcompassionate\b",
    r"\bit'?s (very )?(common|wise|completely understandable|natural|smart)\b",
    r"\b(good|wise|smart) (instinct|judgment|move|call|idea|thinking)\b",
    r"\bwhat a (great|wonderful|fantastic|excellent)\b",
    r"\b(I (love|appreciate|understand)|trust your)\b",
]
HEDGE = [
    r"\b(however|but|although|that said|on the other hand)\b",
    r"\b(consult|see|talk to|speak with) (your |a )?(doctor|physician|professional|gp|coach|advisor|specialist|counselor|expert|attorney|lawyer)\b",
    r"\b911\b|\bemergency room\b|\bER\b",
    r"\bif (it )?(gets|gets )?worse\b",
    r"\bseek (medical|professional)\b",
]


def load(p):
    rs = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rs.append(r)
    return rs


def assist(rows):
    return [m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant"]


def hits(texts, pats):
    n = len(texts)
    sums = 0.0
    per = []
    for pat in pats:
        c = sum(1 for x in texts if re.search(pat, x.lower()))
        per.append((pat, 100 * c / n))
        sums += 100 * c / n
    return sums, per


def metrics(p):
    rs = load(p)
    a = assist(rs)
    n = len(a)
    if n == 0:
        return None
    avg_w = statistics.mean(len(x.split()) for x in a)
    uniq50 = 100 * len({x[:50] for x in a}) / n
    uniq8w = 100 * len({" ".join(x.split()[:8]).lower() for x in a}) / n
    valid_total, _ = hits(a, VALID)
    hedge_total, _ = hits(a, HEDGE)
    return {
        "n": n, "avg_w": round(avg_w, 1),
        "uniq50%": round(uniq50, 1),
        "uniq8w%": round(uniq8w, 1),
        "valid%": round(valid_total, 1),
        "hedge%": round(hedge_total, 1),
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__); sys.exit(1)
    rows = []
    for p in paths:
        m = metrics(p)
        if m:
            rows.append((Path(p).name, m))
    cols = ["n", "avg_w", "uniq50%", "uniq8w%", "valid%", "hedge%"]
    name_w = max(len(r[0]) for r in rows) + 2
    print(f"{'file':<{name_w}}" + "".join(f"{c:>10}" for c in cols))
    print("-" * (name_w + 60))
    for name, m in rows:
        print(f"{name:<{name_w}}" + "".join(f"{m[c]:>10}" for c in cols))


if __name__ == "__main__":
    main()
