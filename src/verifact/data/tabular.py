"""Generic loader for CSV / TSV / JSON / JSONL / XLSX datasets.

Used for IndicClaimBuster (access via the authors' Google form), IFND,
CONSTRAINT 2021 and anything else that ships as a table. You say which
columns hold the claim, evidence, text and label.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..labels import FAKE, REAL, normalise_label


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_pairs(path: str | Path, claim_col: str, evidence_col: str, label_col: str,
               lang: str | None = None, lang_col: str | None = None,
               source: str = "custom") -> list[dict]:
    """Load claim/evidence/label rows (e.g. IndicClaimBuster)."""
    df = read_table(path)
    _check_cols(df, [claim_col, evidence_col, label_col] + ([lang_col] if lang_col else []), path)
    rows = []
    for i, r in df.iterrows():
        claim, evidence = r[claim_col], r[evidence_col]
        if pd.isna(claim) or pd.isna(evidence) or pd.isna(r[label_col]):
            continue
        rows.append({
            "id": f"{source}-{i}",
            "claim": str(claim).strip(),
            "evidence": str(evidence).strip(),
            "label": normalise_label(r[label_col]),
            "lang": str(r[lang_col]) if lang_col else (lang or "unk"),
            "source": source,
        })
    return rows


def load_style(path: str | Path, text_col: str, label_col: str,
               fake_values: list[str], lang: str = "unk", source: str = "custom") -> list[dict]:
    """Load text/label rows for the style (fake vs real) baseline.

    ``fake_values`` lists the label values meaning fake (e.g. ["fake", "0"]).
    Everything else counts as real, so check the dataset's README first.
    """
    df = read_table(path)
    _check_cols(df, [text_col, label_col], path)
    fake_set = {str(v).strip().lower() for v in fake_values}
    rows = []
    for i, r in df.iterrows():
        text = r[text_col]
        if pd.isna(text) or pd.isna(r[label_col]):
            continue
        label = FAKE if str(r[label_col]).strip().lower() in fake_set else REAL
        rows.append({"id": f"{source}-{i}", "text": str(text).strip(), "label": label,
                     "lang": lang, "source": source})
    return rows


def pairs_to_style(rows: list[dict]) -> list[dict]:
    """Turn claim-verification rows into claim-only style rows: SUPPORTS -> REAL,
    REFUTES -> FAKE, NEI dropped. Lets you train the style baseline on the same
    claims and show what evidence adds."""
    out = []
    for r in rows:
        if r["label"] == "NOT ENOUGH INFO":
            continue
        out.append({"id": r["id"], "text": r["claim"],
                    "label": REAL if r["label"] == "SUPPORTS" else FAKE,
                    "lang": r.get("lang", "unk"), "source": r.get("source", "custom")})
    return out


def _check_cols(df: pd.DataFrame, cols: list[str], path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: columns {missing} not found. Available: {list(df.columns)}")
