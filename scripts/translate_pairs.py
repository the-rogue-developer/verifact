"""Machine-translate claim/evidence pairs into Hindi (run on a GPU: Colab/Kaggle).

    # 6k FEVER pairs -> Hindi (about 30-40 min on a T4)
    python scripts/translate_pairs.py --input data/processed/fever/train.jsonl \
        --n 6000 --out data/processed/fever_hi/translated.jsonl

    # Hindi claims with English evidence (the cross-lingual setting)
    python scripts/translate_pairs.py --input data/processed/fever/train.jsonl \
        --n 3000 --claims-only --out data/processed/fever_hi/translated_claims.jsonl

Then: python scripts/prepare_data.py hindi --translated data/processed/fever_hi/translated.jsonl ...
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.data.translate import nllb_translator, translate_pairs  # noqa: E402
from verifact.io import read_jsonl, write_jsonl  # noqa: E402


def balanced_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Take an equal number of rows per label, so the Hindi set is balanced."""
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    per = n // max(len(by_label), 1)
    out = []
    for lab_rows in by_label.values():
        rng.shuffle(lab_rows)
        out += lab_rows[:per]
    rng.shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--tgt", default="hin_Deva")
    p.add_argument("--lang", default="hi")
    p.add_argument("--claims-only", action="store_true", help="translate claims, keep English evidence")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rows = balanced_sample(read_jsonl(a.input), a.n, a.seed)
    print(f"translating {len(rows)} rows: {dict(Counter(r['label'] for r in rows))}")
    fn = nllb_translator(a.model, tgt=a.tgt, batch_size=a.batch_size)
    out = translate_pairs(rows, fn, a.lang, translate_evidence=not a.claims_only,
                          progress=lambda done, total: print(f"  {done}/{total}", flush=True))
    write_jsonl(out, a.out)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
