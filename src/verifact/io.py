"""Reading and writing the project's unified JSONL format.

Every prepared dataset is a JSONL file where each line is one example:

    Pair (claim verification) example:
        {"id": "...", "claim": "...", "evidence": "...", "label": "REFUTES",
         "lang": "hi", "source": "xfact"}

    Single-text (style / claim detection) example:
        {"id": "...", "text": "...", "label": "FAKE", "lang": "en", "source": "ifnd"}

    Retrieval corpus passage:
        {"id": "...", "title": "...", "text": "...", "url": "...", "lang": "en",
         "source": "wikipedia"}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_rows(rows: list[dict], dev_frac: float = 0.1, test_frac: float = 0.1,
               seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split rows into train/dev/test."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test = int(n * test_frac)
    n_dev = int(n * dev_frac)
    test = rows[:n_test]
    dev = rows[n_test:n_test + n_dev]
    train = rows[n_test + n_dev:]
    return train, dev, test
