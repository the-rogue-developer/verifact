# VeriFact results log

One row per (model, test set) evaluation. Add new rows at the bottom.
Accuracy and macro-F1 come from `scripts/evaluate.py` (raw JSON in `results/`
or in the Drive `results/` folder for Colab runs).

Never log single-class test sets (e.g. X-Fact Hindi, all false/misleading) as
real results: `evaluate.py` flags them, and 1.0 there only means "always says false".
They are only mentioned in the notes as a bias check (recall of REFUTES).

"two-way" = `evaluate.py --two-way`: NOT ENOUGH INFO is ignored and the model
picks SUPPORTS vs REFUTES (for test sets with no NEI labels, like X-Fact).

| Date | Model | Train data | Test set | n | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-25 | TF-IDF + LR, claim-only (`--mode single`) | X-Fact train, 13 languages | X-Fact test (13 languages) | 2773 | 0.734 | 0.729 | Style/claim-wording baseline, no evidence. `results/tfidf_claim_xfact.json` |
| 2026-11-25 | TF-IDF + LR, claim+evidence (`--mode pair`) | X-Fact train, 13 languages | X-Fact test (13 languages) | 2773 | 0.752 | 0.748 | Evidence adds only +1.8 acc / +1.9 F1. `results/tfidf_pair_xfact.json` |
| 2026-11-25 | TF-IDF + LR, claim-only (`--mode single`) | FEVER train, 60k | FEVER dev | 10000 | 0.576 | 0.578 | Well above chance (0.33) without evidence: FEVER claims carry label cues |
| 2026-11-25 | TF-IDF + LR, claim+evidence (`--mode pair`) | FEVER train, 60k | FEVER dev | 10000 | 0.582 | 0.573 | Evidence barely helps: bag-of-words can't model how claim and evidence relate |
| 2026-11-25 | mDeBERTa-v3 verifier | FEVER train, 20k, 1 epoch (6.5 min, T4) | FEVER dev | 10000 | 0.836 | 0.837 | +25 points over TF-IDF |
| 2026-11-25 | TF-IDF + LR, both modes | FEVER train, 60k | X-Fact test / OOD / Hindi | 2773 / 1436 / 149 | 0.01–0.17 | – | Fails completely: English word and character features don't transfer to other languages |
| 2026-11-25 | mDeBERTa-v3 verifier, 3-way | FEVER train, 20k, 1 epoch | X-Fact test (13 languages) | 2773 | 0.358 | 0.287 | Zero-shot cross-dataset. NEI predictions are always wrong here (no NEI gold) |
| 2026-11-25 | mDeBERTa-v3 verifier, 3-way | FEVER train, 20k, 1 epoch | X-Fact OOD | 1436 | 0.226 | 0.198 | Hindi bias check: REFUTES recall 0.0 on X-Fact Hindi (n=149, all false) |
| 2026-11-25 | mDeBERTa-v3 verifier, two-way | FEVER train, 20k, 1 epoch | X-Fact test (13 languages) | 2773 | 0.421 | 0.341 | Predicted SUPPORTS for 2586/2773 (93%); see Analysis |
| 2026-11-25 | mDeBERTa-v3 verifier, two-way | FEVER train, 20k, 1 epoch | X-Fact OOD | 1436 | 0.256 | 0.226 | Hindi bias check: REFUTES recall 0.027 on X-Fact Hindi |

## Analysis

### In-domain: the transformer verifier works
On FEVER dev the mDeBERTa verifier reaches 0.837 macro-F1 after one epoch on
20k pairs, against 0.57–0.58 for TF-IDF. TF-IDF gains almost nothing from the
evidence (claim-only 0.578 vs claim+evidence 0.573 F1). A bag of words only
sees which words occur, not whether the evidence agrees with the claim. The
claim-only score is still far above chance, which matches the known
claim-only bias in FEVER: refuted claims were written by annotators who
mutated true sentences, often with negations, so the claim alone gives the
label away (Schuster et al., 2019, "Towards Debiasing Fact Verification Models").

### Cross-dataset: the FEVER verifier says SUPPORTS to almost everything
On X-Fact the same model falls to 0.341 macro-F1 (two-way). The reason is
visible in the predicted labels: it said SUPPORTS for 2586 of 2773 test
claims (93%), although 58% of them are false (1622 REFUTES vs 1151 SUPPORTS).
Its two-way accuracy (0.421) is about what "always SUPPORTS" scores (0.415),
and below "always REFUTES" (0.585). On OOD, where 77% of claims are false,
accuracy drops to 0.256. On the all-false Hindi claims it caught almost none
(REFUTES recall 0.0 three-way, 0.027 two-way).

**Likely cause: the lexical overlap heuristic.** X-Fact evidence consists of
web search snippets retrieved with the claim as the query. Many come from
the fact-check article itself or from pages repeating the rumour, so they
**repeat the claim's words**, whether the claim is true or false. In FEVER,
high word overlap between claim and evidence usually means SUPPORTS: the
evidence is a Wikipedia sentence and true claims paraphrase it. The model
has learned the shortcut "the evidence contains my words, so it supports
me" instead of reasoning about meaning. McCoy et al. (2019), "Right for the
Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language
Inference", showed the same thing for NLI models trained on MNLI: they
predict entailment whenever the premise contains all the hypothesis words,
even when the meaning differs.

Contributing factors:
- **Evidence quality:** X-Fact snippets are not curated gold evidence. Many
  are irrelevant or only mention the topic, so the "right" answer is often
  not in the evidence at all.
- **Different kinds of false claims:** FEVER REFUTES claims are mostly
  simple mutations (negation, a swapped entity) that the evidence directly
  contradicts. Real-world misinformation is usually misleading rather than
  directly contradicted, and the snippet rarely states "this is false".
- **Language shift:** 12 of the 13 X-Fact languages are not English, and
  the model only saw English during fine-tuning.
- **NEI in 3-way:** the model also answers NEI for many claims with weak
  evidence, which is always wrong on X-Fact. That is why two-way scores are higher.

**How to test the overlap explanation (next step):** compute the fraction
of claim tokens that appear in the evidence for each X-Fact example, and
plot P(SUPPORTS) against it, separately for true and false claims. If the
heuristic is the cause, P(SUPPORTS) rises with overlap for *both* classes.

**What this means for VeriFact:** an NLI verifier trained only on FEVER is
not enough for real-world claims. Stage 2 (Hindi / translated data) and
adding real fact-check data are needed. In the pipeline, retrieving
fact-check articles, which usually state the verdict explicitly, matters
more than retrieving pages that merely repeat the claim.
