"""Check a test set for claims that also appear in training (data leakage),
and write a cleaned test file without them.

    python scripts/check_overlap.py
    python scripts/check_overlap.py --test data/processed/hindi/translated_test.jsonl \
        --train data/processed/hindi/train.jsonl --out data/processed/hindi/translated_test.clean.jsonl

A test row counts as leaked if its Hindi claim OR its original English claim
(``orig_claim``, kept by translate_pairs.py) matches any claim or orig_claim
in the training file. Comparing orig_claim catches both a second translation
of the same claim and the English original mixed in with --mix-fever.
Claims are compared after trimming, collapsing whitespace and lower-casing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.io import read_jsonl, write_jsonl  # noqa: E402


def norm(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def claim_keys(row: dict) -> set[str]:
    return {k for k in (norm(row.get("claim")), norm(row.get("orig_claim"))) if k}


def find_overlap(test: list[dict], train: list[dict]) -> dict:
    """Split test rows into leaked / clean and count how they matched."""
    train_claims = {norm(r.get("claim")) for r in train} - {""}
    train_orig = {norm(r.get("orig_claim")) for r in train} - {""}
    train_any = train_claims | train_orig

    leaked, clean = [], []
    by_claim = by_orig = 0
    for r in test:
        hit_claim = norm(r.get("claim")) in train_any
        hit_orig = bool(r.get("orig_claim")) and norm(r.get("orig_claim")) in train_any
        by_claim += hit_claim
        by_orig += hit_orig
        (leaked if hit_claim or hit_orig else clean).append(r)
    distinct = lambda rows: len({norm(r.get("orig_claim") or r.get("claim")) for r in rows})  # noqa: E731
    return {
        "leaked": leaked, "clean": clean,
        "n_test": len(test), "n_leaked": len(leaked), "n_clean": len(clean),
        "match_hindi_claim": by_claim, "match_orig_claim": by_orig,
        "distinct_test_claims": distinct(test), "distinct_leaked_claims": distinct(leaked),
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", default="data/processed/hindi/translated_test.jsonl")
    p.add_argument("--train", default="data/processed/hindi/train.jsonl")
    p.add_argument("--out", default=None, help="cleaned test file (default: <test>.clean.jsonl)")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    test, train = read_jsonl(a.test), read_jsonl(a.train)
    r = find_overlap(test, train)
    pct = lambda k: f"{r[k]:5d}  ({r[k] / max(r['n_test'], 1):.1%})"  # noqa: E731
    print(f"test rows:                 {r['n_test']:5d}   ({r['distinct_test_claims']} distinct claims)")
    print(f"train rows:                {len(train):5d}")
    print(f"Hindi claim in train:      {pct('match_hindi_claim')}")
    print(f"orig_claim in train:       {pct('match_orig_claim')}")
    print(f"leaked (either):           {pct('n_leaked')}   ({r['distinct_leaked_claims']} distinct claims)")
    print(f"clean:                     {pct('n_clean')}")
    print(f"labels before: {dict(Counter(x['label'] for x in test))}")
    print(f"labels after:  {dict(Counter(x['label'] for x in r['clean']))}")

    out = Path(a.out) if a.out else Path(a.test).with_suffix(".clean.jsonl")
    write_jsonl(r["clean"], out)
    print(f"wrote {r['n_clean']} rows to {out}")
    if r["n_leaked"]:
        print("Scores on the original test file are inflated; report the cleaned one, or "
              "re-run prepare_data.py hindi (it now splits by orig_claim) and retrain.")
    return {k: v for k, v in r.items() if k not in ("leaked", "clean")}


if __name__ == "__main__":
    main()
