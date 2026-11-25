"""Build the BM25 and FAISS indexes for the corpus in configs/pipeline.yaml.

    python scripts/build_index.py --config configs/pipeline.yaml
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.config import load_config  # noqa: E402
from verifact.retrieval.bm25 import BM25Retriever  # noqa: E402
from verifact.retrieval.corpus import load_corpus  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    rcfg = load_config(a.config, a.set)["retrieval"]
    passages = load_corpus(rcfg["corpus"])
    out = Path(rcfg.get("index_dir", "artifacts/index"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(passages)} passages")

    if rcfg.get("use_bm25", True):
        t = time.time()
        BM25Retriever(passages).save(out / "bm25.pkl")
        print(f"BM25 index built in {time.time() - t:.1f}s")

    if rcfg.get("use_dense", True):
        from verifact.retrieval.dense import DenseRetriever, sentence_transformer_encoder

        t = time.time()
        enc = sentence_transformer_encoder(rcfg["dense_model"])
        DenseRetriever(passages, enc, rcfg.get("query_prefix", "query: "),
                       rcfg.get("passage_prefix", "passage: ")).save(out / "dense.faiss")
        print(f"FAISS index built in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
