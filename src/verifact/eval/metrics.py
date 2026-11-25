"""Evaluation metrics for verification and retrieval."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(gold: list[str], pred: list[str], labels: list[str] | None = None) -> dict:
    """Accuracy, macro-F1 (over ``labels``), per-class report and confusion matrix.

    The per-class report and confusion matrix cover every label in gold or
    pred, so stray predictions (e.g. NEI on a 2-way test set) stay visible.
    """
    labels = labels or sorted(set(gold) | set(pred))
    all_labels = labels + sorted((set(gold) | set(pred)) - set(labels))
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "per_class": classification_report(gold, pred, labels=all_labels, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": {"labels": all_labels,
                             "matrix": confusion_matrix(gold, pred, labels=all_labels).tolist()},
        "n": len(gold),
        "single_class_test": len(set(gold)) == 1,
    }


def recall_at_k(gold_ids: list[set[str]], retrieved_ids: list[list[str]], k: int) -> float:
    """Fraction of queries where at least one gold passage is in the top k."""
    if not gold_ids:
        return 0.0
    hits = sum(1 for g, r in zip(gold_ids, retrieved_ids) if g and g & set(r[:k]))
    return hits / len(gold_ids)


def mrr(gold_ids: list[set[str]], retrieved_ids: list[list[str]]) -> float:
    """Mean reciprocal rank of the first gold passage."""
    total = 0.0
    for g, r in zip(gold_ids, retrieved_ids):
        for rank, pid in enumerate(r, 1):
            if pid in g:
                total += 1.0 / rank
                break
    return total / max(len(gold_ids), 1)


def fever_style_score(gold_labels: list[str], pred_labels: list[str],
                      gold_ids: list[set[str]], retrieved_ids: list[list[str]], k: int = 5,
                      nei_label: str = "NOT ENOUGH INFO") -> float:
    """Simplified FEVER score: a prediction counts only if the label is right AND
    (for SUPPORTS/REFUTES) a gold evidence passage is in the top-k retrieved.
    For the official number use the FEVER scorer: pip install fever-scorer."""
    correct = 0
    for gl, pl, g, r in zip(gold_labels, pred_labels, gold_ids, retrieved_ids):
        if gl != pl:
            continue
        if gl == nei_label or (g & set(r[:k])):
            correct += 1
    return correct / max(len(gold_labels), 1)
