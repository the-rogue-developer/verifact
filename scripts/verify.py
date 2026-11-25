"""Verify one message from the command line.

    python scripts/verify.py "सरकार 1 दिसंबर से सभी 500 के नोट बंद कर रही है"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.config import load_config  # noqa: E402
from verifact.pipeline.build import build_pipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("message")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    pipe = build_pipeline(load_config(a.config, a.set))
    v = pipe.verify(a.message)
    print(f"\nVERDICT: {v.verdict}  (confidence {v.confidence:.2f}, language {v.language})")
    if v.note:
        print(v.note)
    for c in v.claims:
        print(f"\n  claim: {c.claim}\n  -> {c.label} ({c.confidence:.2f})")
        if c.best_evidence:
            print(f"     evidence: {c.best_evidence['text'][:200]}")
            if c.best_evidence.get("url"):
                print(f"     source:   {c.best_evidence['url']}")
    print("\n" + json.dumps(v.to_dict(), ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
