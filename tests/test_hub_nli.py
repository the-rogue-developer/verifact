"""Off-the-shelf Hub NLI models have no verifact_task.json and use MNLI label
names. Verifier and evaluate.py must map them onto our labels and assume a
pair task. Uses the tiny BERT fixture relabelled like an MNLI model (offline)."""

import importlib.util
import json
import shutil

import pytest

from verifact.labels import NEI, NLI_LABELS, REFUTES, SUPPORTS
from verifact.models.classifier import Verifier, load_task_meta

ROWS = [{"claim": "Ranchi is the capital of Jharkhand.", "evidence": "Ranchi is the capital of Jharkhand.",
         "label": SUPPORTS, "lang": "en", "source": "test"}]


def _relabel(src, dst, id2label):
    shutil.copytree(src, dst)
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["id2label"] = {str(i): lab for i, lab in id2label.items()}
    cfg["label2id"] = {lab: i for i, lab in id2label.items()}
    cfg_path.write_text(json.dumps(cfg))
    return dst


@pytest.fixture
def mnli_model_dir(tiny_model_dir, tmp_path):
    # MNLI order (as in e.g. roberta-large-mnli), deliberately unlike ours.
    return _relabel(tiny_model_dir, tmp_path / "mnli", {0: "contradiction", 1: "neutral", 2: "entailment"})


def test_verifier_normalises_mnli_labels(mnli_model_dir):
    assert load_task_meta(mnli_model_dir) == {}
    v = Verifier(mnli_model_dir, device="cpu")
    # Order must follow the model's output ids, so each prob lands on the right label.
    assert v.labels == [REFUTES, NEI, SUPPORTS]
    probs = v.predict([ROWS[0]["claim"]], [ROWS[0]["evidence"]])
    assert set(probs[0]) == set(NLI_LABELS)


def test_verifier_rejects_unmappable_labels(tiny_model_dir, tmp_path):
    d = _relabel(tiny_model_dir, tmp_path / "bad", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(ValueError, match="can't be mapped"):
        Verifier(d, device="cpu")


def test_evaluate_defaults_to_pair_task(mnli_model_dir, sample_dir):
    spec = importlib.util.spec_from_file_location("evaluate_script", sample_dir.parents[1] / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    preds, labels = evaluate.predict(str(mnli_model_dir), ROWS, batch_size=8)
    assert labels == [REFUTES, NEI, SUPPORTS]
    assert preds[0] in NLI_LABELS
