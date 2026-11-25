# VeriFact results log

One row per (model, test set) evaluation. Add new rows at the bottom.
Accuracy and macro-F1 come from `scripts/evaluate.py` (raw JSON in `results/`).

Never log single-class test sets (e.g. X-Fact Hindi, all false/misleading) as
real results: `evaluate.py` flags them, and 1.0 there only means "always says false".

| Date | Model | Train data | Test set | n | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-25 | TF-IDF + LR, claim-only (`--mode single`) | X-Fact train, 13 languages | X-Fact test (13 languages) | 2773 | 0.734 | 0.729 | Style/claim-wording baseline, no evidence. `results/tfidf_claim_xfact.json` |
| 2026-11-25 | TF-IDF + LR, claim+evidence (`--mode pair`) | X-Fact train, 13 languages | X-Fact test (13 languages) | 2773 | 0.752 | 0.748 | Evidence adds only +1.8 acc / +1.9 F1. `results/tfidf_pair_xfact.json` |
