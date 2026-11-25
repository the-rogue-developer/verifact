# VeriFact: Evidence-Based Multilingual Fake News Detection

VeriFact checks claims in **English, Hindi and Hinglish** messages the way a human
fact-checker does. It finds the factual claims, retrieves evidence from trusted
sources, and uses a fine-tuned transformer to decide whether the evidence
**supports** or **refutes** each claim, then shows that evidence as proof.

```
message ─► clean ─► language ID ─► claim extraction
                                        │  (per claim)
                                        ▼
            BM25 + dense (FAISS) retrieval ─► RRF fusion ─► cross-encoder re-rank
                                        │  top-k passages
                                        ▼
        NLI verifier (mDeBERTa):  [CLS] claim [SEP] evidence [SEP] ─► SUPPORTS / REFUTES / NEI
                                        │
                                        ▼
         rank-weighted aggregation ─► TRUE / FALSE / PARTLY VERIFIED / UNVERIFIED
                                       (+ style-classifier fallback when no evidence is found)
```

## Repository layout

```
src/verifact/
  data/          dataset loaders: FEVER (HF Hub), X-Fact, tables (IndicClaimBuster, IFND), MT
  preprocess/    cleaning, language ID (en / hi / hi-Latn), claim extraction
  baselines/     TF-IDF + logistic regression
  models/        transformer fine-tuning (NLI, style BERT, claim detector) + inference
  retrieval/     corpus chunking, BM25, dense FAISS, RRF hybrid, cross-encoder re-ranker
  pipeline/      end-to-end pipeline + verdict aggregation
  eval/          accuracy, macro-F1, Recall@k, MRR, FEVER-style score
scripts/         prepare_data, fetch_factchecks, translate_pairs, train, train_tfidf,
                 evaluate, build_index, verify
configs/         YAML configs for every model + the pipeline
app/             Streamlit demo
notebooks/       Colab quickstart (GPU training)
tests/           27 offline tests, including a real (tiny-model) training run
data/sample/     small sample corpus and NLI data for smoke tests
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # 27 tests, runs offline on CPU in seconds
```

Training needs a GPU: use `notebooks/colab_quickstart.ipynb` on Google Colab or
Kaggle (free T4). Everything else (TF-IDF baselines, BM25, tests) runs on a laptop.

## Data

| Dataset | Use | How to get it |
|---|---|---|
| FEVER (gold evidence) | Stage-1 verifier training | `scripts/prepare_data.py fever` (Hugging Face Hub; run with `--inspect` first) |
| X-Fact | Multilingual train/test, OOD test | `git clone https://github.com/utahnlp/x-fact data/raw/x-fact` |
| IndicClaimBuster | Hindi / Bengali / code-mixed verification | Password-protected: request access through the Google form in [the repo](https://github.com/pritampal98/indic-claim-buster), then `prepare_data.py hindi --icb FILE --icb-claim-col ...` |
| Machine-translated FEVER | Balanced Hindi training data | `scripts/translate_pairs.py` (NLLB-200) |
| IFND | Style baseline (Indian news) | Hugging Face dataset `rathore11/IndianNews` |
| Fact-check articles | Retrieval corpus | `scripts/fetch_factchecks.py` (Google Fact Check Tools API, free key) |

**Important finding about X-Fact Hindi.** All 1,500+ Hindi X-Fact claims come from two
debunking sites (aajtak.in, hindi.newschecker.in), so **none are labelled true**.
A model trained on all languages learns the shortcut "Hindi → false" and scores a
meaningless 100% on the Hindi test. `evaluate.py` flags single-class test sets.
Balanced Hindi evaluation needs IndicClaimBuster or translated FEVER (with a
human-checked sample).

## Pipeline: commands in order

```bash
# 1. Data
python scripts/prepare_data.py fever --inspect
python scripts/prepare_data.py fever
git clone --depth 1 https://github.com/utahnlp/x-fact data/raw/x-fact

# 2. Baselines
python scripts/train_tfidf.py --mode pair   --train data/processed/fever/train.jsonl --out artifacts/tfidf_pair.joblib
python scripts/train_tfidf.py --mode single --train data/processed/fever/train.jsonl --out artifacts/tfidf_claim.joblib

# 3. Verifier, stage 1 (English FEVER, GPU)
python scripts/train.py --config configs/nli_fever.yaml

# 4. Hindi data + verifier, stage 2
python scripts/translate_pairs.py --input data/processed/fever/train.jsonl --n 6000 \
    --out data/processed/fever_hi/translated.jsonl
python scripts/prepare_data.py hindi --xfact-dir data/raw/x-fact --langs all \
    --translated data/processed/fever_hi/translated.jsonl \
    --mix-fever data/processed/fever/train.jsonl --mix-n 3000
python scripts/train.py --config configs/nli_hindi.yaml

# 5. Evaluate everything on every test set (the cross-dataset table)
python scripts/evaluate.py --model artifacts/nli_hindi/best --data \
    data/processed/hindi/test.jsonl data/processed/hindi/xfact_ood.jsonl \
    data/processed/hindi/translated_test.jsonl --out results/nli_hindi.json

# 6. Retrieval corpus + indexes, then run the system
python scripts/prepare_data.py corpus --docs data/raw/factchecks_hi.jsonl data/raw/factchecks_en.jsonl \
    --from-pairs data/processed/hindi/train.jsonl
python scripts/build_index.py --config configs/pipeline.yaml
python scripts/verify.py "सरकार 1 दिसंबर से सभी 500 के नोट बंद कर रही है"
streamlit run app/streamlit_app.py
```

Quick check without training anything: `--set verifier.model_dir=...` points the
pipeline at any trained model, and `retrieval.corpus=data/sample/...` at a corpus.

## Experiments for the report

| # | Experiment | What it shows |
|---|---|---|
| 1 | TF-IDF claim-only vs TF-IDF claim+evidence vs BERT style vs NLI verifier | What evidence and deep learning each add |
| 2 | Train on FEVER → test on X-Fact / IndicClaimBuster | Cross-dataset generalization |
| 3 | English-only verifier vs + translated Hindi vs + IndicClaimBuster | Cross-lingual transfer |
| 4 | BM25 vs dense vs hybrid vs hybrid + re-ranker (Recall@5, MRR) | The retrieval ablation |
| 5 | Native multilingual vs translate-to-English | Pipeline design choice |
| 6 | Paraphrased claims | Robustness |

### Results so far (TF-IDF + LR, X-Fact, 13 languages)

| Model | X-Fact test (n=2773) acc / macro-F1 | X-Fact OOD (n=1436) acc / macro-F1 |
|---|---|---|
| TF-IDF claim only | 0.734 / 0.729 | 0.768 / 0.695 |
| TF-IDF claim + evidence | 0.752 / 0.748 | 0.787 / 0.712 |

Evidence adds only about 2 points with bag-of-words features. Showing that the NLI
verifier makes much better use of evidence is the core claim of the project.

## Practical tips

- **T4 time budget:** mDeBERTa-base, 256 tokens, 60k examples ≈ 1.5–2 h per epoch.
  Use `--set data.max_train=20000` for a first run.
- If mDeBERTa's loss becomes `nan` with fp16, set `train.fp16=false` or switch to XLM-R.
- Save `artifacts/` to Google Drive: Colab deletes local files when the session ends.
- Fix seeds and report mean ± std over 3 seeds for your main results.

## Acknowledgements

FEVER (Thorne et al., 2018), X-Fact (Gupta & Srikumar, 2021), IndicClaimBuster
(Pal et al., 2025), IFND (Sharma & Garg, 2021), MultiClaim (Pikuliak et al., 2023).
Pretrained models: mDeBERTa-v3, MuRIL, multilingual-E5, mMiniLM cross-encoder, NLLB-200.
