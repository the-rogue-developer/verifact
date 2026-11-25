"""Baseline 1: TF-IDF + logistic regression (runs on CPU in seconds).

    # claim-verification baseline (claim + evidence)
    python scripts/train_tfidf.py --mode pair --train data/processed/fever/train.jsonl \
        --out artifacts/tfidf_pair.joblib

    # style baseline (text only)
    python scripts/train_tfidf.py --mode single --train data/processed/style/train.jsonl \
        --out artifacts/tfidf_style.joblib

Evaluate it with scripts/evaluate.py, like the transformer models.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.baselines.tfidf import save, train_tfidf  # noqa: E402
from verifact.io import read_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pair", "single"], required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--C", type=float, default=4.0)
    a = p.parse_args()
    rows = read_jsonl(a.train)[: a.max_train]
    print(f"training on {len(rows)} rows: {dict(Counter(r['label'] for r in rows))}")
    model = train_tfidf(rows, a.mode, C=a.C)
    save(model, a.out, a.mode)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
