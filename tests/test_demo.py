"""The demo forwards must stay consistent with the sample corpus, and
scripts/run_demo.py must run end to end (BM25 only + the tiny offline model)."""

import importlib.util
from collections import Counter

from verifact.io import read_jsonl
from verifact.preprocess.clean import clean_text
from verifact.preprocess.langid import detect_language
from verifact.retrieval.bm25 import BM25Retriever
from verifact.retrieval.corpus import load_corpus


def _rows(sample_dir):
    return read_jsonl(sample_dir / "demo_forwards.jsonl")


def test_demo_file_is_well_formed(sample_dir):
    rows = _rows(sample_dir)
    doc_ids = {d["id"] for d in read_jsonl(sample_dir / "corpus_docs.jsonl")}
    assert Counter(r["lang"] for r in rows) == {"en": 5, "hi": 5, "hi-Latn": 5}
    assert {r["expected_verdict"] for r in rows} <= {"TRUE", "FALSE"}
    assert all(r["expected_evidence"] in doc_ids for r in rows)


def test_demo_languages_and_retrieval(sample_dir):
    # Deterministic checks (no model): if these fail, the demo messages or the
    # corpus drifted apart, and the demo would show the wrong evidence.
    retriever = BM25Retriever(load_corpus(sample_dir / "corpus_docs.jsonl"))
    for r in _rows(sample_dir):
        assert detect_language(clean_text(r["message"])) == r["lang"], r["id"]
        assert retriever.search(r["message"], k=1)[0][0] == r["expected_evidence"], r["id"]


def test_run_demo_end_to_end(sample_dir, tiny_nli_model_dir, tmp_path, capsys, monkeypatch):
    root = sample_dir.parents[1]
    monkeypatch.chdir(root)  # the config's relative paths resolve from the repo root
    spec = importlib.util.spec_from_file_location("run_demo", root / "scripts" / "run_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    results = demo.main(["--set", "retrieval.use_dense=false", "retrieval.use_reranker=false",
                         f"retrieval.index_dir={tmp_path / 'no_index'}",
                         f"verifier.model_dir={tiny_nli_model_dir}", "--out", str(tmp_path / "demo.json")])
    out = capsys.readouterr().out
    assert len(results) == 15
    assert all(r["verdict"] in {"TRUE", "FALSE", "PARTLY VERIFIED", "UNVERIFIED"} for r in results)
    # Retrieval and language ID don't depend on the (random) tiny model.
    s = demo.summarise(results)
    assert s["language"] == (15, 15) and s["evidence"] == (15, 15)
    assert "matches expected" in out and (tmp_path / "demo.json").exists()
