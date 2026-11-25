"""FEVER loader.

Raw FEVER only stores Wikipedia sentence IDs as evidence, so you'd need the full
Wikipedia dump to get evidence text. It's much easier to use a version of FEVER
that already has the evidence text attached, from the Hugging Face Hub.

Default: ``copenlu/fever_gold_evidence``. Hub datasets change over time, so
``prepare_data.py fever --inspect`` prints the schema first, and you can point
the loader at different columns with the arguments below.
"""

from __future__ import annotations

from ..labels import normalise_label


def _flatten_evidence(ev) -> str:
    """Evidence can be a string, a list of strings, or a list of
    [title, sentence_id, text] triples. Return plain text."""
    if ev is None:
        return ""
    if isinstance(ev, str):
        return ev.strip()
    parts = []
    for item in ev:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, (list, tuple)) and item:
            # [title, sent_id, text] -> keep the text; prefix the title for context
            text = str(item[-1])
            title = str(item[0]).replace("_", " ") if len(item) >= 3 else ""
            parts.append(f"{title}: {text}" if title else text)
        elif isinstance(item, dict):
            parts.append(str(item.get("text") or item.get("sentence") or ""))
    return " ".join(p.strip() for p in parts if p and p.strip())


def load_fever_from_hub(name: str = "copenlu/fever_gold_evidence", split: str = "train",
                        claim_col: str = "claim", evidence_col: str = "evidence",
                        label_col: str = "label", limit: int | None = None) -> list[dict]:
    from datasets import load_dataset  # imported lazily: heavy

    ds = load_dataset(name, split=split)
    missing = [c for c in (claim_col, evidence_col, label_col) if c not in ds.column_names]
    if missing:
        raise KeyError(
            f"Columns {missing} not in {name}. Available: {ds.column_names}. "
            "Pass --claim-col/--evidence-col/--label-col to match."
        )
    label_names = None
    feat = ds.features.get(label_col)
    if hasattr(feat, "names"):
        label_names = feat.names  # ClassLabel -> map ints to strings

    rows = []
    for i, r in enumerate(ds):
        if limit and len(rows) >= limit:
            break
        raw = r[label_col]
        if label_names is not None and isinstance(raw, int):
            raw = label_names[raw]
        evidence = _flatten_evidence(r[evidence_col])
        if not evidence:
            continue
        rows.append({
            "id": f"fever-{split}-{i}",
            "claim": str(r[claim_col]).strip(),
            "evidence": evidence,
            "label": normalise_label(raw),
            "lang": "en",
            "source": "fever",
        })
    return rows


def inspect_hub_dataset(name: str, split: str = "train", n: int = 2) -> None:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    print(f"{name} [{split}] rows={len(ds)}")
    print("features:", ds.features)
    for r in ds.select(range(min(n, len(ds)))):
        print({k: (str(v)[:200]) for k, v in r.items()})
