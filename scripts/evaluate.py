"""Evaluate any trained model (TF-IDF .joblib or transformer folder) on one or
more test files. Passing several files gives you the cross-dataset table.

    python scripts/evaluate.py --model artifacts/nli_hindi/best \
        --data data/processed/hindi/test.jsonl data/processed/hindi/xfact_ood.jsonl \
        --out results/nli_hindi.json

    python scripts/evaluate.py --model artifacts/tfidf_style.joblib \
        --data data/processed/style/test.jsonl

    # X-Fact has only true/false gold labels: --two-way ignores P(NOT ENOUGH INFO)
    # there (datasets that do have NEI labels are still scored 3-way)
    python scripts/evaluate.py --model artifacts/nli_fever/best --two-way \
        --data data/processed/fever/dev.jsonl data/processed/xfact/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.eval.metrics import classification_metrics  # noqa: E402
from verifact.io import read_jsonl  # noqa: E402
from verifact.labels import NEI, REFUTES, SUPPORTS  # noqa: E402


def predict_probs(model_path: str, rows: list[dict], batch_size: int) -> tuple[list[dict], list[str]]:
    """Return ({label: probability} per row, label set the model knows)."""
    if model_path.endswith(".joblib"):
        from verifact.baselines.tfidf import example_to_text, load

        model, mode = load(model_path)
        labels = [str(c) for c in model.classes_]
        P = model.predict_proba([example_to_text(r, mode) for r in rows])
        return [dict(zip(labels, row)) for row in P.tolist()], labels

    from verifact.models.classifier import TextClassifier, Verifier, load_task_meta

    # No verifact_task.json means an off-the-shelf Hub model; those are NLI
    # models in practice, so treat it as a (claim, evidence) pair task.
    meta = load_task_meta(model_path)
    if meta.get("type", "pair") == "pair":
        model = Verifier(model_path, batch_size=batch_size)
        probs = model.predict([r["claim"] for r in rows], [r["evidence"] for r in rows])
    else:
        model = TextClassifier(model_path, batch_size=batch_size)
        probs = model.predict([r.get("text") or r["claim"] for r in rows])
    # model.labels, not meta["labels"]: for Hub models it holds the normalised names.
    return probs, model.labels


def decide(probs: list[dict], two_way: bool = False) -> list[str]:
    """Pick the most probable label per row.

    two_way: ignore NOT ENOUGH INFO and pick whichever of SUPPORTS / REFUTES is
    more probable. For test sets with no NEI gold labels (e.g. X-Fact), where
    the question is only "true or false?": a 3-way model that answers NEI is
    otherwise always wrong, which mostly measures how often it hedges.
    """
    if two_way:
        return [SUPPORTS if p.get(SUPPORTS, 0.0) >= p.get(REFUTES, 0.0) else REFUTES for p in probs]
    return [max(p, key=p.get) for p in probs]


def predict(model_path: str, rows: list[dict], batch_size: int,
            two_way: bool = False) -> tuple[list[str], list[str]]:
    """Return (predictions, label set the model knows)."""
    probs, labels = predict_probs(model_path, rows, batch_size)
    return decide(probs, two_way), labels


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--two-way", action="store_true",
                   help="on test sets with no NOT ENOUGH INFO labels, ignore P(NEI) and predict "
                        "whichever of SUPPORTS/REFUTES is higher (e.g. X-Fact)")
    a = p.parse_args(argv)

    report = {}
    print(f"{'dataset':45s} {'n':>6s} {'acc':>7s} {'macroF1':>8s}")
    for path in a.data:
        rows = read_jsonl(path)
        probs, model_labels = predict_probs(a.model, rows, a.batch_size)
        gold = [r["label"] for r in rows]
        # Only where it makes sense: on a test set that has NEI gold labels,
        # dropping NEI would make those rows impossible to get right.
        two_way = (a.two_way and NEI not in gold
                   and SUPPORTS in model_labels and REFUTES in model_labels)
        pred = decide(probs, two_way)
        # Score on the labels that occur in this test set. If a 3-way model
        # predicts NEI on a 2-way test set, it counts as an error.
        labels = sorted(set(gold))
        m = classification_metrics(gold, pred, labels)
        m["predicted_label_counts"] = {lab: pred.count(lab) for lab in model_labels}
        m["two_way"] = two_way
        report[path] = m
        flag = "  <- single-class test set: this is just recall of that class" if m["single_class_test"] else ""
        print(f"{path:45s} {m['n']:6d} {m['accuracy']:7.4f} {m['macro_f1']:8.4f}{flag}")
        mode = "two-way (NEI ignored)" if two_way else (
            "3-way (test set has NEI labels, --two-way not applied)" if a.two_way and NEI in gold else "argmax")
        print(f"    predicted: {m['predicted_label_counts']}   [{mode}]")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"saved {a.out}")
    return report


if __name__ == "__main__":
    main()
