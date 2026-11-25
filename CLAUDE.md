# VeriFact: project notes for Claude Code

B.Tech minor project (BIT Mesra): evidence-based fake news detection for
English, Hindi and Hinglish. A claim is checked by retrieving evidence and
running a fine-tuned NLI verifier, not by classifying writing style.

## Architecture
clean → langid (en / hi / hi-Latn) → claim extraction → hybrid retrieval
(BM25 + multilingual-e5 dense via FAISS, fused with RRF, cross-encoder re-rank)
→ NLI verifier (mDeBERTa-v3, [CLS] claim [SEP] evidence → SUPPORTS / REFUTES /
NOT ENOUGH INFO) → rank-weighted aggregation → TRUE / FALSE / PARTLY VERIFIED /
UNVERIFIED, with a style-classifier fallback when no evidence is found.

- Package code: `src/verifact/` (data, preprocess, baselines, models, retrieval, pipeline, eval)
- CLI entry points: `scripts/` (prepare_data, train, train_tfidf, evaluate, build_index, verify, translate_pairs, fetch_factchecks)
- All training is config-driven: `configs/*.yaml`, overrides via `--set key=value`
- Unified data format: JSONL with `claim`, `evidence`, `label`, `lang`, `source` (see `src/verifact/io.py`)
- Labels live in `src/verifact/labels.py`; never hard-code label strings elsewhere

## Commands
- Tests (offline, CPU, a few seconds): `python -m pytest -q`
- TF-IDF baseline: `python scripts/train_tfidf.py --mode pair --train FILE --out artifacts/x.joblib`
- Evaluate any model on many test sets: `python scripts/evaluate.py --model PATH --data A.jsonl B.jsonl`
- GPU training runs on Colab/Kaggle via `notebooks/colab_quickstart.ipynb`, not locally

## Rules
- Run `python -m pytest -q` after every code change; add a test for new logic.
- Tests must stay offline: no model downloads (see the tiny BERT fixture in `tests/conftest.py`).
- Keep code compatible with transformers 4.46+ and 5.x (e.g. no `warmup_ratio`, use `processing_class`).
- Hindi text: never tokenise with regex `\w` (it splits words at vowel signs); see `retrieval/bm25.py`.
- Don't commit `data/raw`, `data/processed`, `artifacts/` (large); results JSON is fine.
- I'm a student and must explain every component in my viva: explain *why* when you change something non-obvious.

## Known data issues
- X-Fact Hindi claims are all false/misleading (debunking sites only), so models learn
  "Hindi → false". Never report Hindi X-Fact accuracy as a real result;
  `evaluate.py` flags single-class test sets.
- IndicClaimBuster is password-protected (request via the authors' Google form).
- FEVER Hub dataset columns: check with `prepare_data.py fever --inspect` before preparing.

## Results so far (TF-IDF + LR, X-Fact 13 languages, test n=2773)
claim-only acc 0.734 / F1 0.729; claim+evidence acc 0.752 / F1 0.748
