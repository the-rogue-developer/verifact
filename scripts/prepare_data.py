"""Prepare datasets in VeriFact's unified JSONL format.

Subcommands (run with -h for all options):

  fever   English claim verification from the Hugging Face Hub
          python scripts/prepare_data.py fever --inspect          # look at the schema first
          python scripts/prepare_data.py fever

  hindi   Hindi / multilingual verification set, combining any of: X-Fact (Hindi or all 25
          languages), machine-translated FEVER, IndicClaimBuster, and a slice of FEVER
          mixed in so the model doesn't forget NOT ENOUGH INFO
          python scripts/prepare_data.py hindi --xfact-dir data/raw/x-fact --langs all \
              --translated data/processed/fever_hi/translated.jsonl \
              --mix-fever data/processed/fever/train.jsonl --mix-n 3000

  style   Text-only fake/real data for the style baseline
          python scripts/prepare_data.py style --input data/raw/IFND.csv \
              --text-col Statement --label-col Label --fake-values Fake

  corpus  Evidence corpus for retrieval, from documents and/or dataset evidence
          python scripts/prepare_data.py corpus --docs data/raw/factchecks.jsonl \
              --from-pairs data/processed/hindi/train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.io import read_jsonl, split_rows, write_jsonl  # noqa: E402

log = logging.getLogger("prepare_data")


def _summary(name: str, rows: list[dict]) -> None:
    labels = Counter(r["label"] for r in rows)
    langs = Counter(r.get("lang", "unk") for r in rows)
    log.info("%-28s n=%-7d labels=%s langs=%s", name, len(rows), dict(labels), dict(langs))


def cmd_fever(a):
    from verifact.data.fever import inspect_hub_dataset, load_fever_from_hub

    if a.inspect:
        inspect_hub_dataset(a.hf_name, a.train_split)
        return
    out = Path(a.out_dir)
    for split, fname, limit in ((a.train_split, "train", a.limit), (a.dev_split, "dev", a.dev_limit)):
        rows = load_fever_from_hub(a.hf_name, split, a.claim_col, a.evidence_col, a.label_col, limit)
        write_jsonl(rows, out / f"{fname}.jsonl")
        _summary(f"fever/{fname}", rows)


def cmd_hindi(a):
    from verifact.data.tabular import load_pairs
    from verifact.data.xfact import load_xfact

    out = Path(a.out_dir)
    train, dev, test, extra = [], [], [], {}
    held_out_claims: set[str] = set()  # English claims behind translated dev/test rows

    if a.xfact_dir:
        d = Path(a.xfact_dir) / "data" / "x-fact"
        # NOTE: X-Fact's Hindi claims all come from debunking sites, so they're
        # all false or misleading. Use --langs all to get true claims from other
        # languages and test on Hindi (cross-lingual transfer).
        langs = None if a.langs == "all" else a.langs.split(",")
        load = lambda f: load_xfact(d / f, langs, keep_ambiguous=a.keep_ambiguous)  # noqa: E731
        train += load("train.all.tsv")
        dev += load("dev.all.tsv")
        xtest = load("test.all.tsv")
        test += xtest
        extra["xfact_ood"] = load("ood.tsv")  # different fact-checking sites: a free cross-dataset test
        if langs != ["hi"]:
            extra["xfact_test_hi"] = [r for r in xtest if r["lang"] == "hi"]
            extra["xfact_ood_hi"] = [r for r in extra["xfact_ood"] if r["lang"] == "hi"]

    if a.translated:
        tr = read_jsonl(a.translated)
        # Split by the original English claim: the same claim can appear twice
        # (claims-only and full translation), and both copies must stay in one split.
        t_train, t_dev, t_test = split_rows(tr, 0.05, 0.1, seed=a.seed,
                                            group_key=lambda r: r.get("orig_claim") or r["claim"])
        held_out_claims |= {r.get("orig_claim") or r["claim"] for r in t_dev + t_test}
        train += t_train
        dev += t_dev
        extra["translated_test"] = t_test

    if a.icb:
        icb = load_pairs(a.icb, a.icb_claim_col, a.icb_evidence_col, a.icb_label_col,
                         lang_col=a.icb_lang_col, lang=None if a.icb_lang_col else "hi",
                         source="indicclaimbuster")
        icb_train, icb_dev, icb_test = split_rows(icb, 0.1, 0.1, seed=a.seed)
        train += icb_train
        dev += icb_dev
        test += icb_test
        extra["icb_test"] = icb_test  # also saved separately for per-dataset results

    if a.mix_fever:
        # Drop the English originals of translated dev/test claims, so the
        # model hasn't seen a test claim (in English) during training.
        fever = [r for r in read_jsonl(a.mix_fever) if r["claim"] not in held_out_claims]
        random.Random(a.seed).shuffle(fever)
        train += fever[: a.mix_n]
        log.info("mixed in %d FEVER rows (rehearsal)", min(a.mix_n, len(fever)))

    if not train:
        sys.exit("No data: pass --xfact-dir, --icb and/or --translated")
    n_labels = len({r["label"] for r in train})
    if n_labels < 2:
        log.warning("WARNING: training data has only %d label(s). A classifier can't learn from "
                    "this. Add --langs all, --translated or --icb.", n_labels)
    random.Random(a.seed).shuffle(train)
    for name, rows in [("train", train), ("dev", dev), ("test", test), *extra.items()]:
        write_jsonl(rows, out / f"{name}.jsonl")
        _summary(f"{out.name}/{name}", rows)


def cmd_style(a):
    from verifact.data.tabular import load_style, pairs_to_style

    rows = []
    if a.input:
        rows += load_style(a.input, a.text_col, a.label_col, a.fake_values.split(","),
                           lang=a.lang, source=a.source)
    for p in a.from_pairs or []:
        rows += pairs_to_style(read_jsonl(p))
    if not rows:
        sys.exit("No data: pass --input and/or --from-pairs")
    train, dev, test = split_rows(rows, 0.1, 0.1, seed=a.seed)
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        write_jsonl(part, Path(a.out_dir) / f"{name}.jsonl")
        _summary(f"style/{name}", part)


def cmd_corpus(a):
    from verifact.data.tabular import read_table
    from verifact.retrieval.corpus import documents_to_passages

    docs = []
    for path in a.docs or []:
        if path.endswith(".jsonl"):
            docs += read_jsonl(path)
        else:
            docs += read_table(path).fillna("").to_dict("records")
    # Evidence texts from labelled datasets become passages too, so retrieval
    # can be evaluated (gold passage id = evidence hash). Use TRAIN files only
    # if you also evaluate retrieval on test, otherwise you leak the answers.
    for path in a.from_pairs or []:
        for r in read_jsonl(path):
            h = hashlib.md5(r["evidence"].encode("utf-8")).hexdigest()[:16]
            docs.append({"id": f"ev-{h}", "title": "", "text": r["evidence"],
                         "lang": r.get("lang", "unk"), "source": r.get("source", "dataset"), "url": ""})
    unique = {}
    for d in docs:
        d.setdefault("id", hashlib.md5(d.get("text", "").encode("utf-8")).hexdigest()[:16])
        unique.setdefault(str(d["id"]), d)
    passages = documents_to_passages(list(unique.values()), a.chunk_words, a.overlap)
    n = write_jsonl(passages, a.out)
    log.info("corpus: %d documents -> %d passages -> %s", len(unique), n, a.out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fever")
    f.add_argument("--hf-name", default="copenlu/fever_gold_evidence")
    f.add_argument("--train-split", default="train")
    f.add_argument("--dev-split", default="validation")
    f.add_argument("--claim-col", default="claim")
    f.add_argument("--evidence-col", default="evidence")
    f.add_argument("--label-col", default="label")
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--dev-limit", type=int, default=10000)
    f.add_argument("--out-dir", default="data/processed/fever")
    f.add_argument("--inspect", action="store_true", help="print the dataset schema and exit")
    f.set_defaults(fn=cmd_fever)

    h = sub.add_parser("hindi")
    h.add_argument("--xfact-dir", help="path to a clone of github.com/utahnlp/x-fact")
    h.add_argument("--langs", default="hi", help="comma-separated codes, or 'all'")
    h.add_argument("--translated", help="machine-translated pair file from scripts/translate_pairs.py")
    h.add_argument("--keep-ambiguous", action="store_true",
                   help="map partly-true/complicated X-Fact claims to NOT ENOUGH INFO")
    h.add_argument("--icb", help="IndicClaimBuster file (csv/xlsx/json) after you get access")
    h.add_argument("--icb-claim-col", default="claim")
    h.add_argument("--icb-evidence-col", default="evidence")
    h.add_argument("--icb-label-col", default="label")
    h.add_argument("--icb-lang-col", default=None)
    h.add_argument("--mix-fever", help="FEVER train.jsonl to mix in (keeps NEI alive)")
    h.add_argument("--mix-n", type=int, default=3000)
    h.add_argument("--seed", type=int, default=42)
    h.add_argument("--out-dir", default="data/processed/hindi")
    h.set_defaults(fn=cmd_hindi)

    s = sub.add_parser("style")
    s.add_argument("--input")
    s.add_argument("--text-col", default="text")
    s.add_argument("--label-col", default="label")
    s.add_argument("--fake-values", default="fake,FAKE,Fake,0",
                   help="comma-separated label values meaning fake; check your dataset!")
    s.add_argument("--lang", default="unk")
    s.add_argument("--source", default="custom")
    s.add_argument("--from-pairs", nargs="*", help="also derive claim-only rows from pair files")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out-dir", default="data/processed/style")
    s.set_defaults(fn=cmd_style)

    c = sub.add_parser("corpus")
    c.add_argument("--docs", nargs="*", help="document files (jsonl/csv/xlsx) with title,text,url,lang,source")
    c.add_argument("--from-pairs", nargs="*", help="use the evidence texts of these pair files as passages")
    c.add_argument("--chunk-words", type=int, default=120)
    c.add_argument("--overlap", type=int, default=30)
    c.add_argument("--out", default="data/corpus/passages.jsonl")
    c.set_defaults(fn=cmd_corpus)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
