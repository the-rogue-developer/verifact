"""Shared fixtures. ``tiny_model_dir`` builds a randomly initialised 2-layer BERT
with a vocabulary made from the sample data, so the training and inference
code can be tested offline on CPU in seconds. Its predictions are meaningless;
the tests only check that the code runs end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    out = tmp_path_factory.mktemp("tiny_bert")
    texts = []
    for name in ("nli_train.jsonl", "nli_dev.jsonl", "corpus_docs.jsonl"):
        for line in (SAMPLE / name).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            texts += [r.get("claim", ""), r.get("evidence", ""), r.get("text", ""), r.get("title", "")]
    chars = sorted({c for t in texts for c in t.lower() if not c.isspace()})
    words = sorted({w for t in texts for w in t.lower().split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + [f"##{c}" for c in chars] + words
    vocab = list(dict.fromkeys(vocab))
    (out / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")

    tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512, num_labels=3)
    BertForSequenceClassification(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def tiny_nli_model_dir(tiny_model_dir, tmp_path_factory) -> Path:
    """The tiny model with our NLI label names in its config, so Verifier
    accepts it (the raw fixture has placeholder labels LABEL_0..2)."""
    import shutil

    from verifact.labels import NLI_LABELS

    out = tmp_path_factory.mktemp("tiny_nli") / "model"
    shutil.copytree(tiny_model_dir, out)
    cfg = json.loads((out / "config.json").read_text())
    cfg["id2label"] = {str(i): lab for i, lab in enumerate(NLI_LABELS)}
    cfg["label2id"] = {lab: i for i, lab in enumerate(NLI_LABELS)}
    (out / "config.json").write_text(json.dumps(cfg))
    return out
