"""Some X-Fact claims alone are longer than max_length. Pair tokenisation must
truncate them instead of failing with "Sequence to truncate too short to
respect the provided max_length" (what truncation="only_second" does)."""

from verifact.io import read_jsonl, write_jsonl
from verifact.models.classifier import Verifier
from verifact.models.train_classifier import train

LONG_CLAIM = " ".join(["Ranchi is the capital of Jharkhand and the government announced a new scheme"] * 10)
EVIDENCE = "Ranchi is the capital of the Indian state of Jharkhand."


def test_verifier_handles_claim_longer_than_max_length(tiny_nli_model_dir):
    v = Verifier(tiny_nli_model_dir, device="cpu")
    v.max_length = 32
    assert len(v.tokenizer(LONG_CLAIM)["input_ids"]) > v.max_length  # the claim alone is too long
    probs = v.predict([LONG_CLAIM, "Short claim."], [EVIDENCE, EVIDENCE])
    assert len(probs) == 2 and abs(sum(probs[0].values()) - 1) < 1e-4


def test_training_handles_claim_longer_than_max_length(sample_dir, tiny_model_dir, tmp_path):
    rows = read_jsonl(sample_dir / "nli_train.jsonl")
    rows[0] = {**rows[0], "claim": LONG_CLAIM}
    write_jsonl(rows, tmp_path / "train.jsonl")
    results = train({
        "output_dir": str(tmp_path / "run"),
        "task": {"type": "pair"},
        "model": {"name": str(tiny_model_dir), "max_length": 32},
        "data": {"train": str(tmp_path / "train.jsonl"), "dev": str(tmp_path / "train.jsonl")},
        "train": {"epochs": 1, "batch_size": 4, "fp16": False, "logging_steps": 1},
    })
    assert results["dev"]["n"] == len(rows)
