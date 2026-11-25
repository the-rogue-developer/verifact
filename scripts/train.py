"""Fine-tune a transformer from a config.

    python scripts/train.py --config configs/nli_fever.yaml
    python scripts/train.py --config configs/nli_fever.yaml --set train.epochs=1 data.max_train=5000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.config import load_config  # noqa: E402
from verifact.models.train_classifier import train  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=1")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    results = train(cfg)
    print(json.dumps({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in results.items()},
                     indent=2))


if __name__ == "__main__":
    main()
