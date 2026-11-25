"""Fine-tune a pretrained transformer for sequence classification.

One training script for three models, chosen by the config's ``task.type``:
    * ``pair``   - claim verification (NLI): [CLS] claim [SEP] evidence [SEP] -> 3 labels
    * ``single`` - BERT style baseline (text -> REAL/FAKE) or claim detector

How the verifier works: the pretrained encoder reads claim and evidence
together, so self-attention lets every claim token attend to every evidence
token. The final hidden state of the first token ([CLS]) goes through a
linear classification head, trained with (optionally class-weighted)
cross-entropy.

Run:  python scripts/train.py --config configs/nli_fever.yaml
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..eval.metrics import classification_metrics
from ..io import read_jsonl
from ..labels import NLI_LABELS, STYLE_LABELS

log = logging.getLogger(__name__)


def _label_list(cfg: dict) -> list[str]:
    labels = cfg["task"].get("labels")
    if labels:
        return labels
    return NLI_LABELS if cfg["task"]["type"] == "pair" else STYLE_LABELS


def _filter_labels(rows: list[dict], labels: list[str]) -> list[dict]:
    keep = [r for r in rows if r["label"] in labels]
    if len(keep) < len(rows):
        log.warning("Dropped %d rows whose label is not in %s", len(rows) - len(keep), labels)
    return keep


def compute_class_weights(rows: list[dict], labels: list[str]) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1. Helps the rare classes
    (NOT ENOUGH INFO is usually underrepresented)."""
    counts = Counter(r["label"] for r in rows)
    freq = np.array([counts.get(lab, 0) for lab in labels], dtype=np.float64)
    freq = np.where(freq == 0, 1.0, freq)
    w = freq.sum() / (len(labels) * freq)
    return torch.tensor(w / w.mean(), dtype=torch.float)


def train(cfg: dict) -> dict:
    """Train and evaluate. Returns the dev (and, if given, test) metrics."""
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                              TrainingArguments, set_seed)

    task, data, tcfg = cfg["task"], cfg["data"], cfg["train"]
    labels = _label_list(cfg)
    label2id = {lab: i for i, lab in enumerate(labels)}
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(tcfg.get("seed", 42))

    train_rows = _filter_labels(read_jsonl(data["train"]), labels)
    dev_rows = _filter_labels(read_jsonl(data["dev"]), labels)
    if data.get("max_train"):
        train_rows = train_rows[: data["max_train"]]
    log.info("train=%d dev=%d labels=%s", len(train_rows), len(dev_rows), dict(Counter(r["label"] for r in train_rows)))

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), label2id=label2id,
        id2label={i: lab for lab, i in label2id.items()},
        ignore_mismatched_sizes=True,  # lets you start from an already fine-tuned NLI head
    )
    # Keep the master weights in fp32. transformers 5.x loads weights in the
    # checkpoint's own dtype, and mDeBERTa's Hub weights are stored in fp16.
    # Mixed-precision training (fp16=True) needs fp32 weights: it runs the
    # forward pass in fp16 but keeps fp32 weights and gradients so tiny
    # updates aren't rounded to zero. With fp16 weights the gradient scaler
    # fails with "ValueError: Attempting to unscale FP16 gradients".
    model = model.float()
    max_len = cfg["model"].get("max_length", 256)
    is_pair = task["type"] == "pair"

    def to_dataset(rows):
        ds = Dataset.from_list([
            {"a": r["claim"] if is_pair else (r.get("text") or r["claim"]),
             "b": r["evidence"] if is_pair else "",
             "label": label2id[r["label"]]}
            for r in rows
        ])

        def tok(batch):
            if is_pair:
                # "longest_first" removes tokens from whichever text is longer until the
                # pair fits. Usually that is the evidence, so the claim stays whole, as
                # with "only_second". But some X-Fact claims alone exceed max_length, and
                # "only_second" (which may only cut the evidence) then raises an error.
                return tokenizer(batch["a"], batch["b"], truncation="longest_first", max_length=max_len)
            return tokenizer(batch["a"], truncation=True, max_length=max_len)

        return ds.map(tok, batched=True, remove_columns=["a", "b"])

    train_ds, dev_ds = to_dataset(train_rows), to_dataset(dev_rows)

    class_weights = compute_class_weights(train_rows, labels) if tcfg.get("class_weights", True) else None
    if class_weights is not None:
        log.info("class weights: %s", dict(zip(labels, class_weights.tolist())))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            y = inputs.pop("labels")
            outputs = model(**inputs)
            w = class_weights.to(outputs.logits.device) if class_weights is not None else None
            loss = nn.functional.cross_entropy(outputs.logits, y, weight=w)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        pred = np.argmax(logits, axis=-1)
        m = classification_metrics([labels[i] for i in y], [labels[i] for i in pred], labels)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    use_cuda = torch.cuda.is_available()
    # Warmup as an absolute step count: works on transformers 4.x and 5.x
    # (5.x removed the warmup_ratio argument).
    steps_per_epoch = math.ceil(len(train_ds) / (tcfg.get("batch_size", 16) * tcfg.get("grad_accum", 1)))
    warmup_steps = int(steps_per_epoch * tcfg.get("epochs", 3) * tcfg.get("warmup_ratio", 0.06))
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=float(tcfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=tcfg.get("batch_size", 16),
        per_device_eval_batch_size=tcfg.get("eval_batch_size", 64),
        gradient_accumulation_steps=tcfg.get("grad_accum", 1),
        num_train_epochs=tcfg.get("epochs", 3),
        weight_decay=tcfg.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_cuda and tcfg.get("fp16", True),
        logging_steps=tcfg.get("logging_steps", 50),
        report_to=tcfg.get("report_to", "none"),  # set "wandb" to use Weights & Biases
        seed=tcfg.get("seed", 42),
        dataloader_num_workers=tcfg.get("num_workers", 2 if use_cuda else 0),
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tcfg.get("patience", 2))],
    )
    trainer.train()

    best_dir = out_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "verifact_task.json").write_text(json.dumps({"type": task["type"], "labels": labels,
                                                              "max_length": max_len}))

    results = {"dev": _evaluate_rows(trainer, dev_rows, to_dataset, labels)}
    if data.get("test"):
        test_rows = _filter_labels(read_jsonl(data["test"]), labels)
        results["test"] = _evaluate_rows(trainer, test_rows, to_dataset, labels)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info("dev macro-F1=%.4f", results["dev"]["macro_f1"])
    return results


def _evaluate_rows(trainer, rows, to_dataset, labels) -> dict:
    out = trainer.predict(to_dataset(rows))
    pred = np.argmax(out.predictions, axis=-1)
    return classification_metrics([r["label"] for r in rows], [labels[i] for i in pred], labels)
