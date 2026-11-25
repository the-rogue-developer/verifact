"""Inference wrappers around fine-tuned models saved by train_classifier.py."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..labels import normalise_label


def _device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_task_meta(model_dir: str | Path) -> dict:
    """Read verifact_task.json written by train_classifier.py.

    Returns {} when it is missing, e.g. for an off-the-shelf model from the
    Hugging Face Hub (``model_dir`` is then a Hub id like
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", not a local folder).
    """
    meta_path = Path(model_dir) / "verifact_task.json"
    return json.loads(meta_path.read_text()) if meta_path.is_file() else {}


class _Base:
    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = load_task_meta(model_dir)
        # Pass a str, not a Path: on Windows Path("org/model") becomes "org\model",
        # which from_pretrained would no longer recognise as a Hub id.
        model_dir = str(model_dir)
        self.device = _device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # .float(): transformers 5.x keeps the checkpoint's dtype, so a model saved
        # or published in fp16 would otherwise run in fp16, which is slow or
        # unsupported for some ops on CPU and gives different scores than training.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(self.device).eval()
        id2label = self.model.config.id2label
        self.labels = meta.get("labels") or [id2label[i] for i in range(len(id2label))]
        self.max_length = meta.get("max_length", 256)
        self.batch_size = batch_size

    @torch.inference_mode()
    def _probs(self, a: list[str], b: list[str] | None) -> list[dict[str, float]]:
        out = []
        for i in range(0, len(a), self.batch_size):
            aa = a[i:i + self.batch_size]
            kwargs = dict(padding=True, max_length=self.max_length, return_tensors="pt")
            if b is not None:
                # "longest_first", not "only_second": it cuts the longer text, normally the
                # evidence, and still works when a claim alone exceeds max_length
                # (only_second can't shorten the claim and raises an error). Must match training.
                enc = self.tokenizer(aa, b[i:i + self.batch_size], truncation="longest_first", **kwargs)
            else:
                enc = self.tokenizer(aa, truncation=True, **kwargs)
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            for row in torch.softmax(logits.float(), dim=-1).cpu().tolist():
                out.append(dict(zip(self.labels, row)))
        return out


class Verifier(_Base):
    """Claim-verification (NLI) model: (claim, evidence) -> label probabilities.

    Works with our fine-tuned models and with off-the-shelf NLI models whose
    config uses MNLI names (entailment / neutral / contradiction): those are
    mapped onto SUPPORTS / NOT ENOUGH INFO / REFUTES via normalise_label, so the
    pipeline and evaluate.py always see our label names.
    """

    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 32):
        super().__init__(model_dir, device, batch_size)
        try:
            labels = [normalise_label(lab) for lab in self.labels]
        except ValueError as e:
            raise ValueError(f"{model_dir}: model labels {self.labels} can't be mapped to NLI labels ({e})") from e
        if len(set(labels)) != len(labels):
            raise ValueError(f"{model_dir}: model labels {self.labels} map to duplicate NLI labels {labels}")
        self.labels = labels

    def predict(self, claims: list[str], evidences: list[str]) -> list[dict[str, float]]:
        if len(claims) != len(evidences):
            raise ValueError("claims and evidences must be the same length")
        return self._probs(claims, evidences) if claims else []


class TextClassifier(_Base):
    """Single-text classifier: style baseline or claim detector."""

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        return self._probs(texts, None) if texts else []

    def positive_scores(self, texts: list[str], positive_label: str) -> list[float]:
        return [p[positive_label] for p in self.predict(texts)]
