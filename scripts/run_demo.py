"""Run the full pipeline on the sample WhatsApp-style forwards and print a table.

    python scripts/run_demo.py
    python scripts/run_demo.py --set verifier.model_dir=artifacts/nli_fever/best

    # laptop, no GPU: BM25 only, any NLI model folder or Hub id
    python scripts/run_demo.py --set retrieval.use_dense=false retrieval.use_reranker=false \
        verifier.model_dir=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

Takes the same --config / --set options as scripts/verify.py. By default it
searches the sample corpus (data/sample/corpus_docs.jsonl), which is what the
demo messages were written against. Each message in data/sample/demo_forwards.jsonl
records the expected language, verdict and evidence document, so the summary
shows where each stage succeeds or fails: language ID, retrieval, verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.config import load_config  # noqa: E402
from verifact.io import read_jsonl  # noqa: E402
from verifact.pipeline.build import build_pipeline  # noqa: E402

# Applied before the user's --set, so the user can still override them.
# index_dir points at a folder that normally doesn't exist, so indexes are built
# fresh from the sample corpus. Loading artifacts/index (built from the full
# corpus) with the sample passages would make its passage ids point at the wrong text.
DEMO_DEFAULTS = ["retrieval.corpus=data/sample/corpus_docs.jsonl", "retrieval.index_dir=artifacts/index_demo"]


def _width(s: str) -> int:
    # Devanagari vowel signs / virama are combining marks that take no column
    # of their own, so len() over-counts and the table columns drift.
    return sum(0 if unicodedata.category(c) in ("Mn", "Me", "Cf") else 1 for c in s)


def _cell(s: str, width: int) -> str:
    s = " ".join(str(s).split())
    if _width(s) > width:
        out = ""
        for c in s:
            if _width(out + c) > width - 1:
                break
            out += c
        s = out + "…"
    return s + " " * (width - _width(s))


COLUMNS = [("#", 3), ("lang", 8), ("message", 38), ("claim", 38), ("verdict", 16), ("conf", 5),
           ("evidence", 30), ("expected", 9)]


def _doc_id(passage_id: str | None) -> str | None:
    # Passages built with documents_to_passages are named "<doc id>#<chunk>".
    return passage_id.split("#")[0] if passage_id else None


def run(pipe, rows: list[dict]) -> list[dict]:
    results = []
    for row in rows:
        v = pipe.verify(row["message"])
        top = v.claims[0] if v.claims else None
        ev = top.best_evidence if top else None
        results.append({
            "id": row.get("id"), "message": row["message"],
            "lang": v.language, "expected_lang": row.get("lang"),
            "claim": top.claim if top else "", "claim_label": top.label if top else None,
            "n_claims": len(v.claims),
            "verdict": v.verdict, "confidence": v.confidence, "expected_verdict": row.get("expected_verdict"),
            "evidence_id": ev.get("id") if ev else None, "evidence_title": ev.get("title", "") if ev else "",
            "expected_evidence": row.get("expected_evidence"),
        })
    return results


def print_table(results: list[dict]) -> None:
    print(" ".join(_cell(name, w) for name, w in COLUMNS))
    print(" ".join("-" * w for _, w in COLUMNS))
    for i, r in enumerate(results, 1):
        cells = [i, r["lang"], r["message"], r["claim"], r["verdict"], f"{r['confidence']:.2f}",
                 r["evidence_title"] or "-", r["expected_verdict"] or "-"]
        print(" ".join(_cell(c, w) for c, (_, w) in zip(cells, COLUMNS)))
    if any(r["n_claims"] > 1 for r in results):
        print("\n(only the first claim of each message is shown; the verdict covers all of them)")


def summarise(results: list[dict]) -> dict:
    def score(key, expected_key, fn=lambda x: x):
        pairs = [(fn(r[key]), r[expected_key]) for r in results if r[expected_key]]
        return sum(a == b for a, b in pairs), len(pairs)

    return {
        "language": score("lang", "expected_lang"),
        "evidence": score("evidence_id", "expected_evidence", _doc_id),
        "verdict": score("verdict", "expected_verdict"),
    }


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/sample/demo_forwards.jsonl")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, as in verify.py")
    p.add_argument("--out", default=None, help="also save the per-message results as JSON")
    a = p.parse_args(argv)

    # Windows consoles default to a legacy code page that can't print Devanagari.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(a.config, DEMO_DEFAULTS + a.set)
    model_dir = cfg["verifier"]["model_dir"]
    try:
        pipe = build_pipeline(cfg)
    except OSError as e:
        sys.exit(f"Could not load the verifier from {model_dir!r} ({e}).\n"
                 "Train one first, or point to a model: --set verifier.model_dir=PATH_OR_HUB_ID")

    rows = read_jsonl(a.data)
    results = run(pipe, rows)
    print(f"\nverifier: {model_dir}   corpus: {cfg['retrieval']['corpus']}   messages: {len(rows)}\n")
    print_table(results)
    s = summarise(results)
    print("\nmatches expected:  " + "   ".join(f"{k} {ok}/{n}" for k, (ok, n) in s.items()))
    print("(language = language ID, evidence = retrieval found the intended document, "
          "verdict = the whole pipeline got it right)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": s, "results": results}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"saved {a.out}")
    return results


if __name__ == "__main__":
    main()
