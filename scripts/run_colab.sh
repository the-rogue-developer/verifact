#!/usr/bin/env bash
# Today's Colab run: FEVER + X-Fact data, both TF-IDF baselines, mDeBERTa NLI
# fine-tune (20k rows, 1 epoch), and evaluation on FEVER dev and X-Fact test.
#
# Run from the repo root, after mounting Google Drive:
#     bash scripts/run_colab.sh "/content/drive/MyDrive/verifact_run"
#
# Everything you want to keep (models + results JSON) is written to that folder,
# so it survives the Colab session ending.

set -euo pipefail   # stop on the first error, on unset variables, and on failures inside pipes

if [ $# -ne 1 ]; then
    echo "usage: bash scripts/run_colab.sh OUTPUT_DIR   (e.g. a Google Drive path)" >&2
    exit 1
fi
OUT="$1"
mkdir -p "$OUT/models" "$OUT/results"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "1/7 Prepare FEVER"
python scripts/prepare_data.py fever

step "2/7 Clone X-Fact into data/raw/x-fact"
if [ -d data/raw/x-fact/.git ]; then
    echo "already cloned, skipping"
else
    git clone --depth 1 https://github.com/utahnlp/x-fact data/raw/x-fact
fi

step "3/7 Prepare X-Fact (all languages) into data/processed/xfact"
python scripts/prepare_data.py hindi --xfact-dir data/raw/x-fact --langs all --out-dir data/processed/xfact

step "4/7 Train TF-IDF baselines on FEVER (max 60000)"
python scripts/train_tfidf.py --mode pair --train data/processed/fever/train.jsonl \
    --max-train 60000 --out "$OUT/models/tfidf_pair.joblib"
python scripts/train_tfidf.py --mode single --train data/processed/fever/train.jsonl \
    --max-train 60000 --out "$OUT/models/tfidf_claim_only.joblib"

step "5/7 Fine-tune NLI verifier (configs/nli_fever.yaml, 20000 rows, 1 epoch)"
# Train on the local disk, not on Drive: the trainer's checkpoints include
# optimizer state (several GB for mDeBERTa) and writing them to Drive is slow
# and eats the free quota. Only the final model is copied to Drive below.
python scripts/train.py --config configs/nli_fever.yaml \
    --set data.max_train=20000 train.epochs=1 output_dir=artifacts/nli_fever

step "6/7 Copy NLI model and training results to $OUT"
rm -rf "$OUT/models/nli_fever"
cp -r artifacts/nli_fever/best "$OUT/models/nli_fever"
cp artifacts/nli_fever/results.json "$OUT/results/nli_fever_train.json"

step "7/7 Evaluate all models on FEVER dev and X-Fact test"
for name in tfidf_pair.joblib tfidf_claim_only.joblib nli_fever; do
    echo "--- $name"
    python scripts/evaluate.py --model "$OUT/models/$name" \
        --data data/processed/fever/dev.jsonl data/processed/xfact/test.jsonl \
        --out "$OUT/results/eval_${name%.joblib}.json"
done

step "Done. Models in $OUT/models, results in $OUT/results"
ls -R "$OUT"
