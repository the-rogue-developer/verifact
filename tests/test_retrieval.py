import numpy as np

from verifact.io import read_jsonl
from verifact.retrieval.bm25 import BM25Retriever, tokenize
from verifact.retrieval.corpus import chunk_text, documents_to_passages
from verifact.retrieval.dense import DenseRetriever
from verifact.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _passages(sample_dir):
    return documents_to_passages(read_jsonl(sample_dir / "corpus_docs.jsonl"))


def test_tokenize_keeps_hindi_words_whole():
    toks = tokenize("सरकार 500 के नोट बंद कर रही है।")
    assert "सरकार" in toks and "500" in toks and "नोट" in toks
    assert "के" not in toks and "है" not in toks  # stopwords


def test_chunking():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, chunk_words=100, overlap=20)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w80"
    assert chunk_text("short text") == ["short text"]


def test_bm25_finds_relevant_passage(sample_dir, tmp_path):
    passages = _passages(sample_dir)
    r = BM25Retriever(passages)
    assert r.search("Is Ranchi the capital of Jharkhand?", k=3)[0][0] == "doc-ranchi#0"
    assert r.search("500 रुपये के नोट बंद", k=3)[0][0] == "doc-rbi-500-hi#0"
    r.save(tmp_path / "bm25.pkl")
    r2 = BM25Retriever.load(tmp_path / "bm25.pkl", passages)
    assert r2.search("Taj Mahal Agra", k=1)[0][0] == "doc-taj#0"


def _bag_of_words_encoder(vocab):
    """Stand-in for a sentence embedding model: normalised bag-of-words vectors."""
    def encode(texts):
        out = np.zeros((len(texts), len(vocab)), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                if tok in vocab:
                    out[i, vocab[tok]] += 1
        return out
    return encode


def test_dense_retriever_with_faiss(sample_dir, tmp_path):
    passages = _passages(sample_dir)
    vocab = {}
    for p in passages:
        for t in tokenize(p["title"] + " " + p["text"]):
            vocab.setdefault(t, len(vocab))
    enc = _bag_of_words_encoder(vocab)
    d = DenseRetriever(passages, enc, query_prefix="", passage_prefix="")
    assert d.search("lunar south pole landing", k=2)[0][0] == "doc-chandrayaan#0"
    d.save(tmp_path / "dense.faiss")
    d2 = DenseRetriever.load(tmp_path / "dense.faiss", passages, enc, "", "")
    assert d2.search("boiling point of water", k=1)[0][0] == "doc-water#0"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[("a", 9), ("b", 5), ("c", 1)], [("b", 0.9), ("d", 0.8), ("a", 0.1)]])
    assert [pid for pid, _ in fused][:2] == ["b", "a"]


def test_hybrid_returns_passage_dicts(sample_dir):
    passages = _passages(sample_dir)
    h = HybridRetriever(passages, [BM25Retriever(passages)])
    hits = h.search("Rs 2000 notes withdrawn", k=2)
    assert hits[0]["id"] == "doc-2000#0"
    assert "score" in hits[0] and "url" in hits[0]
