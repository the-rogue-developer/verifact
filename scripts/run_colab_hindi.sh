#!/usr/bin/env bash
# Stage 2 on Colab: adapt the FEVER verifier to Hindi using the machine-translated
# FEVER pairs from scripts/run_kaggle_translate.sh, and compare it with the
# zero-shot FEVER-only model.
#
# Run from the repo root, after mounting Google Drive:
#     bash scripts/run_colab_hindi.sh /content/drive/MyDrive/verifact/translated \
#                                     /content/drive/MyDrive/verifact
#
#   arg 1: Drive folder with fever_hi_claims_only.jsonl and fever_hi_full.jsonl
#   arg 2: Drive output folder (the same one given to run_colab.sh). The Hindi model
#          goes to models/nli_hindi, every evaluation JSON to results/.

set -euo pipefail   # stop on the first error, on unset variables, and on failures inside pipes

if [ $# -ne 2 ]; then
    echo "usage: bash scripts/run_colab_hindi.sh TRANSLATED_DIR OUTPUT_DIR" >&2
    exit 1
fi
TRANSLATED_DIR="$1"
OUT="$2"
RESULTS="$OUT/results"
HINDI_DIR="$OUT/models/nli_hindi"
mkdir -p "$RESULTS" data/processed/fever_hi

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

TRANSLATED=data/processed/fever_hi/translated.jsonl
TRANSLATED_TEST=data/processed/hindi/translated_test.jsonl
FEVER_DEV=data/processed/fever/dev.jsonl
XFACT_HI=data/processed/xfact/xfact_test_hi.jsonl
TEST_SETS=("$TRANSLATED_TEST" "$FEVER_DEV" "$XFACT_HI")

# ---------------------------------------------------------------------------
step "1/6 Combine translated files into $TRANSLATED"
if [ ! -f data/processed/fever/train.jsonl ] || [ ! -f "$FEVER_DEV" ]; then
    echo "FEVER not prepared yet, preparing it first"
    python scripts/prepare_data.py fever
fi
for f in fever_hi_claims_only.jsonl fever_hi_full.jsonl; do
    [ -f "$TRANSLATED_DIR/$f" ] || { echo "missing $TRANSLATED_DIR/$f" >&2; exit 1; }
done
# Python rather than cat: a file without a final newline would glue two rows together.
python - "$TRANSLATED" "$TRANSLATED_DIR/fever_hi_claims_only.jsonl" "$TRANSLATED_DIR/fever_hi_full.jsonl" <<'EOF'
import json, sys
from collections import Counter
out, *inputs = sys.argv[1:]
rows = [json.loads(line) for path in inputs for line in open(path, encoding="utf-8") if line.strip()]
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"rows: {len(rows)}")
print(f"labels: {dict(Counter(r['label'] for r in rows))}")
print(f"langs: {dict(Counter(r['lang'] for r in rows))}")
print(f"distinct claims: {len({r.get('orig_claim') or r['claim'] for r in rows})}")
EOF

# ---------------------------------------------------------------------------
step "2/6 Prepare Hindi training data (translated FEVER + 2000 English FEVER rows, no X-Fact)"
# X-Fact is deliberately left out: its Hindi claims are all false/misleading, so the
# model would learn "Hindi -> false". The English FEVER rows keep NOT ENOUGH INFO
# alive and reduce forgetting of English.
python scripts/prepare_data.py hindi --translated "$TRANSLATED" \
    --mix-fever data/processed/fever/train.jsonl --mix-n 2000

# X-Fact Hindi is only used as an extra test set (prepared in its own folder).
if [ ! -f "$XFACT_HI" ]; then
    echo "X-Fact Hindi test set not found, preparing it (test only)"
    [ -d data/raw/x-fact/.git ] || git clone --depth 1 https://github.com/utahnlp/x-fact data/raw/x-fact
    python scripts/prepare_data.py hindi --xfact-dir data/raw/x-fact --langs all --out-dir data/processed/xfact
fi

# ---------------------------------------------------------------------------
step "3/6 Zero-shot baseline: FEVER-only model on the test sets"
if [ -f artifacts/nli_fever/best/verifact_task.json ]; then
    FEVER_MODEL=artifacts/nli_fever/best
else
    # Python instead of find | head, which can trip pipefail. Skip the Hindi model
    # (it also has a verifact_task.json) and trainer checkpoints; prefer nli_fever.
    FEVER_MODEL=$(python - "$OUT" <<'EOF'
import sys
from pathlib import Path
found = [p.parent for p in Path(sys.argv[1]).rglob("verifact_task.json")
         if "nli_hindi" not in p.parts and "checkpoints" not in p.parts]
found.sort(key=lambda d: ("nli_fever" not in str(d), str(d)))
print(found[0] if found else "")
EOF
)
fi
if [ -z "$FEVER_MODEL" ]; then
    echo "No FEVER model found in artifacts/nli_fever/best or under $OUT. Run run_colab.sh first." >&2
    exit 1
fi
echo "FEVER model: $FEVER_MODEL"
# Evaluated on all three sets (not only translated_test) so step 6 can compare
# like with like. --two-way only affects X-Fact Hindi, which has no NEI labels.
python scripts/evaluate.py --model "$FEVER_MODEL" --two-way --data "${TEST_SETS[@]}" \
    --out "$RESULTS/hindi_zeroshot.json"

# ---------------------------------------------------------------------------
step "4/6 Train configs/nli_hindi.yaml from the FEVER model (2 epochs) -> $HINDI_DIR"
# data.test: the config's default test.jsonl is empty without X-Fact/IndicClaimBuster.
python scripts/train.py --config configs/nli_hindi.yaml --set \
    "model.name=$FEVER_MODEL" train.epochs=2 "output_dir=$HINDI_DIR" \
    "data.test=$TRANSLATED_TEST"
cp "$HINDI_DIR/results.json" "$RESULTS/nli_hindi_train.json"
# The trainer's checkpoints (weights + optimizer state, several GB) are only needed
# during training; the final model is in $HINDI_DIR/best. Free the Drive space.
rm -rf "$HINDI_DIR/checkpoints"

# ---------------------------------------------------------------------------
step "5/6 Evaluate the Hindi model (translated test, FEVER dev for forgetting, X-Fact Hindi --two-way)"
python scripts/evaluate.py --model "$HINDI_DIR/best" --two-way --data "${TEST_SETS[@]}" \
    --out "$RESULTS/hindi_finetuned.json"

# ---------------------------------------------------------------------------
step "6/6 Comparison: zero-shot FEVER model vs Hindi model"
python - "$RESULTS/hindi_zeroshot.json" "$RESULTS/hindi_finetuned.json" <<'EOF' | tee "$RESULTS/hindi_comparison.txt"
import json, sys
zs, ft = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:])
print(f"{'test set':42s} {'n':>6s} | {'zero-shot acc':>13s} {'F1':>6s} | {'Hindi acc':>9s} {'F1':>6s} | {'dF1':>6s}")
print("-" * 100)
for path in zs:
    a, b = zs[path], ft[path]
    note = ("  (two-way)" if b.get("two_way") else "") + \
           ("  SINGLE-CLASS: only recall of that class, not a real result" if b["single_class_test"] else "")
    print(f"{path:42s} {b['n']:6d} | {a['accuracy']:13.4f} {a['macro_f1']:6.4f} | "
          f"{b['accuracy']:9.4f} {b['macro_f1']:6.4f} | {b['macro_f1'] - a['macro_f1']:+6.3f}{note}")
EOF

step "Done. Hindi model: $HINDI_DIR/best   results: $RESULTS"
ls "$RESULTS"
