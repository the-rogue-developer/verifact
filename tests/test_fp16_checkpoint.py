"""Hub checkpoints stored in fp16 (like microsoft/mdeberta-v3-base) must be
trained and served with fp32 weights. transformers 5.x keeps the checkpoint's
dtype on load; fp16 master weights crash mixed-precision training on GPU with
"Attempting to unscale FP16 gradients"."""

import torch
from safetensors import safe_open

from verifact.models.classifier import Verifier
from verifact.models.train_classifier import train


def _fp16_copy(src, dst):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoModelForSequenceClassification.from_pretrained(src).half().save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    with safe_open(str(dst / "model.safetensors"), "pt") as f:
        assert f.get_tensor(next(iter(f.keys()))).dtype == torch.float16  # really an fp16 checkpoint
    return dst


def test_verifier_loads_fp16_checkpoint_as_fp32(tiny_nli_model_dir, tmp_path):
    v = Verifier(_fp16_copy(tiny_nli_model_dir, tmp_path / "fp16"), device="cpu")
    assert all(p.dtype == torch.float32 for p in v.model.parameters())


def test_training_from_fp16_checkpoint_keeps_fp32_weights(sample_dir, tiny_model_dir, tmp_path):
    start = _fp16_copy(tiny_model_dir, tmp_path / "fp16")
    train({
        "output_dir": str(tmp_path / "run"),
        "task": {"type": "pair"},
        "model": {"name": str(start), "max_length": 64},
        "data": {"train": str(sample_dir / "nli_train.jsonl"), "dev": str(sample_dir / "nli_dev.jsonl")},
        "train": {"epochs": 1, "batch_size": 4, "fp16": False, "logging_steps": 1},
    })
    with safe_open(str(tmp_path / "run" / "best" / "model.safetensors"), "pt") as f:
        assert all(f.get_tensor(k).dtype == torch.float32 for k in f.keys())
