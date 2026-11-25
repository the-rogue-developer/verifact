"""Hybrid retrieval (BM25 + dense) fused with Reciprocal Rank Fusion, then an
optional cross-encoder re-ranker.

RRF: score(d) = sum over retrievers of 1 / (k + rank_r(d)). It only uses
ranks, so it doesn't matter that BM25 and cosine scores are on different
scales. k=60 is the value from the original RRF paper.

Re-ranking: a cross-encoder reads (query, passage) *together*, like the
verifier, so it's far more accurate than a bi-encoder but too slow to run
on the whole corpus. We run it only on the fused top ~50.
"""

from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]: ...


def reciprocal_rank_fusion(result_lists: list[list[tuple[str, float]]], k: int = 60,
                           weights: list[float] | None = None) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        for rank, (pid, _) in enumerate(results, 1):
            fused[pid] = fused.get(pid, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """candidates: list of (passage_id, passage_text)."""
        if not candidates:
            return []
        scores = self.model.predict([(query, text) for _, text in candidates], batch_size=self.batch_size)
        return sorted(((pid, float(s)) for (pid, _), s in zip(candidates, scores)), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, passages: list[dict], retrievers: list[Retriever],
                 weights: list[float] | None = None, reranker: CrossEncoderReranker | None = None,
                 candidates: int = 50, rrf_k: int = 60):
        from .corpus import passage_text

        self.by_id = {p["id"]: p for p in passages}
        self._text = passage_text
        self.retrievers = retrievers
        self.weights = weights
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k passages as dicts with an added 'score'."""
        lists = [r.search(query, self.candidates) for r in self.retrievers]
        fused = reciprocal_rank_fusion(lists, k=self.rrf_k, weights=self.weights)[: self.candidates]
        if self.reranker is not None:
            fused = self.reranker.rerank(query, [(pid, self._text(self.by_id[pid])) for pid, _ in fused])
        return [{**self.by_id[pid], "score": score} for pid, score in fused[:k]]
