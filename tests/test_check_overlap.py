"""scripts/check_overlap.py finds test claims that also occur in training,
by Hindi claim or by orig_claim, and writes a cleaned test file."""

import importlib.util
from pathlib import Path

from verifact.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("check_overlap", ROOT / "scripts" / "check_overlap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(i, claim, orig=None, label="SUPPORTS"):
    r = {"id": i, "claim": claim, "evidence": "e", "label": label, "lang": "hi", "source": "t"}
    if orig:
        r["orig_claim"] = orig
    return r


def test_check_overlap_finds_and_removes_leaked_claims(tmp_path, capsys):
    train = [
        _row("t1", "दावा एक", "claim one"),          # translated (claims-only copy)
        _row("t2", "Claim  Two", None),               # English FEVER row mixed in, odd spacing/case
        _row("t3", "दावा तीन", "claim three"),
    ]
    test = [
        _row("x1", "दावा एक", "claim one"),          # same Hindi claim -> leaked
        _row("x2", "दावा दो (अलग अनुवाद)", "claim two"),  # different translation, English original in train
        _row("x3", "दावा चार", "claim four", "REFUTES"),  # clean
        _row("x4", "दावा पाँच", "claim five", "REFUTES"),  # clean
    ]
    write_jsonl(train, tmp_path / "train.jsonl")
    write_jsonl(test, tmp_path / "test.jsonl")

    r = _load().main(["--test", str(tmp_path / "test.jsonl"), "--train", str(tmp_path / "train.jsonl")])
    assert r["n_leaked"] == 2 and r["n_clean"] == 2
    assert r["match_hindi_claim"] == 1 and r["match_orig_claim"] == 2
    clean = read_jsonl(tmp_path / "test.clean.jsonl")
    assert [x["id"] for x in clean] == ["x3", "x4"]
    assert "leaked (either)" in capsys.readouterr().out


def test_no_overlap_after_prepare_hindi(tmp_path):
    """End to end: data prepared with the grouped split has nothing to clean."""
    import subprocess
    import sys

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    # FEVER itself repeats claims (same claim, different evidence): 30 claims x 2 rows.
    fever = [_row(f"f{i}-{j}", f"claim {i}", None, labels[i % 3]) for i in range(30) for j in range(2)]
    for r in fever:
        r["lang"] = "en"
    translated = [{**r, "id": f"{r['id']}-mt-{lang}", "claim": f"दावा {r['claim']}", "orig_claim": r["claim"],
                   "lang": lang} for r in fever for lang in ("hi-claim/en-evidence", "hi")]
    write_jsonl(fever, tmp_path / "fever.jsonl")
    write_jsonl(translated, tmp_path / "translated.jsonl")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "hindi",
                    "--translated", str(tmp_path / "translated.jsonl"),
                    "--mix-fever", str(tmp_path / "fever.jsonl"), "--mix-n", "60",
                    "--out-dir", str(tmp_path / "hindi")], check=True, capture_output=True)
    test, train = (read_jsonl(tmp_path / "hindi" / f) for f in ("translated_test.jsonl", "train.jsonl"))
    r = _load().find_overlap(test, train)
    assert r["n_test"] > 0 and r["n_leaked"] == 0
