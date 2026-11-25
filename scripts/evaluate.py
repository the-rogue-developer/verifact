"""Evaluate any trained model (TF-IDF .joblib or transformer folder) on one or
more test files. Passing several files gives you the cross-dataset table.

    python scripts/evaluate.py --model artifacts/nli_hindi/best \
        --data data/processed/hindi/test.jsonl data/processed/hindi/xfact_ood.jsonl \
        --out results/nli_hindi.json

    python scripts/evaluate.py --model artifacts/tfidf_style.joblib \
        --data data/processed/style/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.eval.metrics import classification_metrics  # noqa: E402
from verifact.io import read_jsonl  # noqa: E402


def predict(model_path: str, rows: list[dict], batch_size: int) -> tuple[list[str], list[str]]:
    """Return (predictions, label set the model knows)."""
    if model_path.endswith(".joblib"):
        from verifact.baselines.tfidf import load, predict_tfidf

        model, mode = load(model_path)
        return predict_tfidf(model, rows, mode), list(model.classes_)

    meta = json.loads((Path(model_path) / "verifact_task.json").read_text())
    if meta["type"] == "pair":
        from verifact.models.classifier import Verifier

        v = Verifier(model_path, batch_size=batch_size)
        probs = v.predict([r["claim"] for r in rows], [r["evidence"] for r in rows])
    else:
        from verifact.models.classifier import TextClassifier

        c = TextClassifier(model_path, batch_size=batch_size)
        probs = c.predict([r.get("text") or r["claim"] for r in rows])
    return [max(p, key=p.get) for p in probs], meta["labels"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    report = {}
    print(f"{'dataset':45s} {'n':>6s} {'acc':>7s} {'macroF1':>8s}")
    for path in a.data:
        rows = read_jsonl(path)
        pred, model_labels = predict(a.model, rows, a.batch_size)
        gold = [r["label"] for r in rows]
        # Score on the labels that occur in this test set. If a 3-way model
        # predicts NEI on a 2-way test set, it counts as an error.
        labels = sorted(set(gold))
        m = classification_metrics(gold, pred, labels)
        m["predicted_label_counts"] = {lab: pred.count(lab) for lab in model_labels}
        report[path] = m
        flag = "  <- single-class test set: this is just recall of that class" if m["single_class_test"] else ""
        print(f"{path:45s} {m['n']:6d} {m['accuracy']:7.4f} {m['macro_f1']:8.4f}{flag}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
