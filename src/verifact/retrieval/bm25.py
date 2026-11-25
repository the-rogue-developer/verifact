"""Sparse retrieval with BM25 (keyword matching).

BM25 is good at exact matches: names, numbers, rare words. The tokenizer
splits on whitespace and punctuation rather than using regex ``\\w``, which
would break Hindi words at vowel signs (matras).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .corpus import passage_text

_TOKEN = re.compile(r"[^\s.,!?।;:\"'()\[\]{}<>|/\\\-–—_*#@&%+=~`^$]+")

# A small stopword list; keep it tiny so names and numbers survive.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or", "for",
    "on", "at", "by", "with", "that", "this", "it", "as", "be", "from",
    "है", "हैं", "का", "की", "के", "को", "में", "से", "और", "ने", "पर", "यह", "था", "थी",
}


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, passages: list[dict], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.ids = [p["id"] for p in passages]
        self.bm25 = BM25Okapi([tokenize(passage_text(p)) or ["<empty>"] for p in passages], k1=k1, b=b)

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str | Path, passages: list[dict]) -> "BM25Retriever":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        r = cls.__new__(cls)
        r.passages, r.ids, r.bm25 = passages, obj["ids"], obj["bm25"]
        if r.ids != [p["id"] for p in passages]:
            raise ValueError("BM25 index does not match the corpus; rebuild it")
        return r
