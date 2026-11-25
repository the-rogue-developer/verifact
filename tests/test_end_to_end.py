"""Offline end-to-end test: train the tiny model with the real training code,
load it with the real inference wrapper, and run the full pipeline with BM25."""

import json

from verifact.baselines.tfidf import load, predict_tfidf, save, train_tfidf
from verifact.io import read_jsonl
from verifact.labels import NLI_LABELS
from verifact.models.classifier import Verifier
from verifact.models.train_classifier import compute_class_weights, train
from verifact.pipeline.pipeline import VeriFactPipeline
from verifact.retrieval.bm25 import BM25Retriever
from verifact.retrieval.corpus import documents_to_passages
from verifact.retrieval.hybrid import HybridRetriever


def test_class_weights_upweight_rare_class():
    rows = [{"label": "SUPPORTS"}] * 8 + [{"label": "REFUTES"}] * 2
    w = compute_class_weights(rows, ["SUPPORTS", "REFUTES"])
    assert w[1] > w[0]


def test_tfidf_baseline(sample_dir, tmp_path):
    rows = read_jsonl(sample_dir / "nli_train.jsonl")
    model = train_tfidf(rows, "pair")
    save(model, tmp_path / "m.joblib", "pair")
    m2, mode = load(tmp_path / "m.joblib")
    preds = predict_tfidf(m2, read_jsonl(sample_dir / "nli_dev.jsonl"), mode)
    assert set(preds) <= set(NLI_LABELS)


def test_train_verify_pipeline(sample_dir, tiny_model_dir, tmp_path):
    cfg = {
        "output_dir": str(tmp_path / "run"),
        "task": {"type": "pair"},
        "model": {"name": str(tiny_model_dir), "max_length": 128},
        "data": {"train": str(sample_dir / "nli_train.jsonl"), "dev": str(sample_dir / "nli_dev.jsonl"),
                 "test": str(sample_dir / "nli_dev.jsonl")},
        "train": {"epochs": 1, "batch_size": 4, "learning_rate": 1e-3, "fp16": False, "logging_steps": 1},
    }
    results = train(cfg)
    assert {"dev", "test"} <= set(results)
    best = tmp_path / "run" / "best"
    assert json.loads((best / "verifact_task.json").read_text())["labels"] == NLI_LABELS

    verifier = Verifier(best, device="cpu")
    probs = verifier.predict(["Ranchi is the capital of Jharkhand."], ["Ranchi is the capital of Jharkhand."])
    assert set(probs[0]) == set(NLI_LABELS) and abs(sum(probs[0].values()) - 1) < 1e-4

    passages = documents_to_passages(read_jsonl(sample_dir / "corpus_docs.jsonl"))
    pipe = VeriFactPipeline(HybridRetriever(passages, [BM25Retriever(passages)]), verifier)
    for msg in ["Forwarded as received!! RBI will ban all Rs 500 notes from 1 December. Share with everyone!",
                "सरकार 1 दिसंबर से सभी 500 रुपये के नोट बंद कर रही है, जल्दी बैंक जाएं!"]:
        v = pipe.verify(msg)
        assert v.verdict in {"TRUE", "FALSE", "PARTLY VERIFIED", "UNVERIFIED"}
        assert v.claims and v.claims[0].evidence, "retrieval should find evidence for the note-ban claim"
        assert "500" in v.claims[0].evidence[0].passage["text"]
        json.dumps(v.to_dict(), ensure_ascii=False)  # serialisable for the API/UI
