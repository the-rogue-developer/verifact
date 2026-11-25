"""Dense retrieval: multilingual sentence embeddings + FAISS.

A bi-encoder embeds the query and every passage *separately* into vectors;
retrieval is nearest-neighbour search by cosine similarity (inner product
on L2-normalised vectors). Passages are embedded once, offline, so search
is fast. Because the embedding model is multilingual, a Hindi claim can
retrieve an English fact-check (cross-lingual retrieval).

E5 models expect "query: " / "passage: " prefixes; other models (e.g. LaBSE)
use none. Set them in configs/pipeline.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .corpus import passage_text

Encoder = Callable[[list[str]], np.ndarray]


def sentence_transformer_encoder(model_name: str, device: str | None = None,
                                 batch_size: int = 64) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=len(texts) > 1000)

    return encode


class DenseRetriever:
    def __init__(self, passages: list[dict], encoder: Encoder,
                 query_prefix: str = "query: ", passage_prefix: str = "passage: ",
                 index=None):
        import faiss

        self.passages = passages
        self.ids = [p["id"] for p in passages]
        self.encoder = encoder
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        if index is None:
            vecs = self._normalise(self.encoder([passage_prefix + passage_text(p) for p in passages]))
            index = faiss.IndexFlatIP(vecs.shape[1])  # exact search; fine up to ~1M passages
            index.add(vecs)
        self.index = index

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype="float32")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norms, 1e-12, None)

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        q = self._normalise(self.encoder([self.query_prefix + query]))
        scores, idx = self.index.search(q, min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def save(self, path: str | Path) -> None:
        import faiss
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, passages: list[dict], encoder: Encoder,
             query_prefix: str = "query: ", passage_prefix: str = "passage: ") -> "DenseRetriever":
        import faiss
        index = faiss.read_index(str(path))
        if index.ntotal != len(passages):
            raise ValueError("FAISS index size does not match the corpus; rebuild it")
        return cls(passages, encoder, query_prefix, passage_prefix, index=index)
