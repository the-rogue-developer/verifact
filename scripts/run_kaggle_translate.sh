#!/usr/bin/env bash
# Today's Kaggle run: machine-translate FEVER pairs into Hindi with NLLB-200
# (needs the GPU accelerator turned on).
#
# Run from the repo root:
#     bash scripts/run_kaggle_translate.sh            # writes to /kaggle/working
#     bash scripts/run_kaggle_translate.sh OUT_DIR    # or somewhere else
#
# Files in /kaggle/working are kept as the notebook's output when you
# "Save Version", so download them from there afterwards.

set -euo pipefail   # stop on the first error, on unset variables, and on failures inside pipes

OUT="${1:-/kaggle/working}"
mkdir -p "$OUT"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "1/3 Prepare FEVER"
python scripts/prepare_data.py fever

step "2/3 Translate 3000 pairs, claims only (Hindi claim + English evidence)"
python scripts/translate_pairs.py --input data/processed/fever/train.jsonl \
    --n 3000 --claims-only --out "$OUT/fever_hi_claims_only.jsonl"

step "3/3 Translate 1000 full pairs (Hindi claim + Hindi evidence)"
python scripts/translate_pairs.py --input data/processed/fever/train.jsonl \
    --n 1000 --out "$OUT/fever_hi_full.jsonl"

step "Done"
wc -l "$OUT"/fever_hi_*.jsonl
