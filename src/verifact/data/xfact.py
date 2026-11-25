"""X-Fact loader (multilingual real-world fact-checks, includes Hindi).

Get the data with:
    git clone --depth 1 https://github.com/utahnlp/x-fact data/raw/x-fact

Files live in data/raw/x-fact/data/x-fact/{train.all,dev.all,test.all,ood,zeroshot}.tsv
with columns: language, site, evidence_1..evidence_5, link_1..link_5, claimDate,
reviewDate, claimant, claim, label.

X-Fact has 7 veracity labels. We map them to our 3-way scheme:
    true, mostly true                  -> SUPPORTS
    false, mostly false                -> REFUTES
    partly true/misleading, complicated -> dropped by default (ambiguous),
                                           or NOT ENOUGH INFO with keep_ambiguous=True
    other                              -> dropped

Caveat for your report: X-Fact "evidence" is web search snippets, not curated
gold evidence, so some snippets are irrelevant. That makes it a harder,
more realistic test set than FEVER.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from ..labels import NEI, REFUTES, SUPPORTS

XFACT_LABEL_MAP = {
    "true": SUPPORTS,
    "mostly true": SUPPORTS,
    "false": REFUTES,
    "mostly false": REFUTES,
}
XFACT_AMBIGUOUS = {"partly true/misleading", "complicated/hard to categorise"}


def load_xfact(path: str | Path, languages: list[str] | None = None,
               keep_ambiguous: bool = False, max_snippets: int = 5) -> list[dict]:
    """Load one X-Fact TSV file into unified pair examples.

    Args:
        path: path to an X-Fact .tsv file.
        languages: ISO codes to keep (e.g. ["hi"]); None keeps all.
        keep_ambiguous: map partly-true/complicated claims to NOT ENOUGH INFO
            instead of dropping them.
        max_snippets: how many evidence snippets to join into the evidence text.
    """
    csv.field_size_limit(sys.maxsize)
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for i, r in enumerate(reader):
            lang = (r.get("language") or "").strip()
            if languages and lang not in languages:
                continue
            raw_label = (r.get("label") or "").strip().lower()
            if raw_label in XFACT_LABEL_MAP:
                label = XFACT_LABEL_MAP[raw_label]
            elif keep_ambiguous and raw_label in XFACT_AMBIGUOUS:
                label = NEI
            else:
                continue
            snippets = [(r.get(f"evidence_{k}") or "").strip() for k in range(1, max_snippets + 1)]
            snippets = [s for s in snippets if s]
            claim = (r.get("claim") or "").strip()
            if not claim or not snippets:
                continue
            rows.append({
                "id": f"xfact-{path.stem}-{i}",
                "claim": claim,
                "evidence": " ".join(snippets),
                "label": label,
                "lang": lang,
                "source": "xfact",
                "site": (r.get("site") or "").strip(),
            })
    return rows
