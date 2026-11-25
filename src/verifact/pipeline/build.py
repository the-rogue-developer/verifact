"""Build a VeriFactPipeline from configs/pipeline.yaml."""

from __future__ import annotations

from pathlib import Path

from ..labels import FAKE
from ..preprocess.claims import ClaimExtractor
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.corpus import load_corpus
from ..retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from .pipeline import PipelineSettings, VeriFactPipeline


def build_retriever(cfg: dict, passages: list[dict] | None = None) -> HybridRetriever:
    rcfg = cfg["retrieval"]
    passages = passages if passages is not None else load_corpus(rcfg["corpus"])
    index_dir = Path(rcfg.get("index_dir", "artifacts/index"))
    retrievers, weights = [], []

    if rcfg.get("use_bm25", True):
        bm25_path = index_dir / "bm25.pkl"
        retrievers.append(BM25Retriever.load(bm25_path, passages) if bm25_path.exists()
                          else BM25Retriever(passages))
        weights.append(rcfg.get("bm25_weight", 1.0))

    if rcfg.get("use_dense", True):
        from ..retrieval.dense import DenseRetriever, sentence_transformer_encoder

        enc = sentence_transformer_encoder(rcfg["dense_model"])
        qp, pp = rcfg.get("query_prefix", "query: "), rcfg.get("passage_prefix", "passage: ")
        faiss_path = index_dir / "dense.faiss"
        retrievers.append(DenseRetriever.load(faiss_path, passages, enc, qp, pp) if faiss_path.exists()
                          else DenseRetriever(passages, enc, qp, pp))
        weights.append(rcfg.get("dense_weight", 1.0))

    if not retrievers:
        raise ValueError("Enable at least one of use_bm25 / use_dense")

    reranker = CrossEncoderReranker(rcfg["reranker_model"]) if rcfg.get("use_reranker") else None
    return HybridRetriever(passages, retrievers, weights, reranker,
                           candidates=rcfg.get("candidates", 50))


def build_pipeline(cfg: dict) -> VeriFactPipeline:
    from ..models.classifier import TextClassifier, Verifier

    retriever = build_retriever(cfg)
    verifier = Verifier(cfg["verifier"]["model_dir"])

    extractor = ClaimExtractor(max_claims=cfg.get("claims", {}).get("max_claims", 5))
    cd = cfg.get("claim_detector") or {}
    if cd.get("model_dir"):
        clf = TextClassifier(cd["model_dir"])
        pos = cd.get("positive_label", "CHECKWORTHY")
        extractor.scorer = lambda sents: clf.positive_scores(sents, pos)
        extractor.threshold = cd.get("threshold", 0.5)

    style_scorer = None
    st = cfg.get("style") or {}
    if st.get("model_dir"):
        style_clf = TextClassifier(st["model_dir"])
        style_scorer = lambda text: style_clf.positive_scores([text], FAKE)[0]

    p = cfg.get("pipeline", {})
    settings = PipelineSettings(
        top_k_evidence=p.get("top_k_evidence", 3),
        claim_threshold=p.get("claim_threshold", 0.6),
        rank_decay=p.get("rank_decay", 0.7),
        style_threshold=p.get("style_threshold", 0.7),
    )
    return VeriFactPipeline(retriever, verifier, extractor, style_scorer, None, settings)
