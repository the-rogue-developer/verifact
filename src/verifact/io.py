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
from typing import Callable, Iterable, Iterator


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
               seed: int = 42, group_key: Callable[[dict], str] | None = None
               ) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split rows into train/dev/test.

    group_key: rows with the same key always land in the same split. Use it
    when one claim appears in several rows (e.g. translated with English and
    with Hindi evidence): otherwise the model sees a test claim in training
    and the test score is inflated (data leakage).
    """
    rows = list(rows)
    if group_key is None:
        random.Random(seed).shuffle(rows)
        n = len(rows)
        n_test = int(n * test_frac)
        n_dev = int(n * dev_frac)
        test = rows[:n_test]
        dev = rows[n_test:n_test + n_dev]
        train = rows[n_test + n_dev:]
        return train, dev, test

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_test, n_dev = int(len(rows) * test_frac), int(len(rows) * dev_frac)
    train, dev, test = [], [], []
    for k in keys:
        # Fill test, then dev, up to their target row counts; the rest is train.
        target = test if len(test) < n_test else dev if len(dev) < n_dev else train
        target += groups[k]
    return train, dev, test
