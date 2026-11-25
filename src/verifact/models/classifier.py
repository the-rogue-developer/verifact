"""Inference wrappers around fine-tuned models saved by train_classifier.py."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Base:
    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        meta_path = model_dir / "verifact_task.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.device = _device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
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
                enc = self.tokenizer(aa, b[i:i + self.batch_size], truncation="only_second", **kwargs)
            else:
                enc = self.tokenizer(aa, truncation=True, **kwargs)
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            for row in torch.softmax(logits.float(), dim=-1).cpu().tolist():
                out.append(dict(zip(self.labels, row)))
        return out


class Verifier(_Base):
    """Claim-verification (NLI) model: (claim, evidence) -> label probabilities."""

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
