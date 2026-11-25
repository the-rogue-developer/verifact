"""--two-way in scripts/evaluate.py: on test sets without NOT ENOUGH INFO gold
labels, ignore P(NEI) and pick SUPPORTS vs REFUTES; leave 3-way sets alone."""

import importlib.util
from pathlib import Path

import pytest

from verifact.baselines.tfidf import save, train_tfidf
from verifact.io import read_jsonl, write_jsonl
from verifact.labels import NEI, REFUTES, SUPPORTS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evaluate():
    spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decide_ignores_nei_only_when_two_way(evaluate):
    probs = [{SUPPORTS: 0.2, REFUTES: 0.3, NEI: 0.5}, {SUPPORTS: 0.4, REFUTES: 0.1, NEI: 0.5}]
    assert evaluate.decide(probs) == [NEI, NEI]
    assert evaluate.decide(probs, two_way=True) == [REFUTES, SUPPORTS]


@pytest.fixture
def test_sets(sample_dir, tmp_path):
    rows = read_jsonl(sample_dir / "nli_dev.jsonl")
    two = tmp_path / "two_way.jsonl"
    write_jsonl([r for r in rows if r["label"] != NEI], two)
    return str(sample_dir / "nli_dev.jsonl"), str(two)


@pytest.fixture
def models(sample_dir, tiny_nli_model_dir, tmp_path):
    tfidf = tmp_path / "tfidf.joblib"
    save(train_tfidf(read_jsonl(sample_dir / "nli_train.jsonl"), "pair"), tfidf, "pair")
    return {"tfidf": str(tfidf), "transformer": str(tiny_nli_model_dir)}


@pytest.mark.parametrize("kind", ["tfidf", "transformer"])
def test_two_way_flag(evaluate, models, test_sets, kind, capsys):
    three, two = test_sets
    report = evaluate.main(["--model", models[kind], "--data", three, two, "--two-way"])
    out = capsys.readouterr().out

    # 3-way test set: flag not applied, scored as usual.
    assert report[three]["two_way"] is False
    # 2-way test set: NEI is never predicted.
    assert report[two]["two_way"] is True
    counts = report[two]["predicted_label_counts"]
    assert counts[NEI] == 0 and counts[SUPPORTS] + counts[REFUTES] == report[two]["n"]
    # Predicted label counts are printed for every dataset.
    assert out.count("predicted:") == 2 and "two-way (NEI ignored)" in out


def test_without_flag_nothing_changes(evaluate, models, test_sets):
    _, two = test_sets
    report = evaluate.main(["--model", models["tfidf"], "--data", two])
    assert report[two]["two_way"] is False
