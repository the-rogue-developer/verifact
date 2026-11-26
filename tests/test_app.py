"""The Streamlit app, driven headless with streamlit.testing (offline: BM25 only,
tiny random verifier). Checks the example picker and a full "Check" click."""

from pathlib import Path

import pytest
import yaml

from verifact.io import read_jsonl

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


@pytest.fixture
def app_config(tiny_nli_model_dir, tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg["retrieval"].update(corpus=str(ROOT / "data" / "sample" / "corpus_docs.jsonl"),
                            index_dir=str(tmp_path / "no_index"), use_dense=False, use_reranker=False)
    cfg["verifier"]["model_dir"] = str(tiny_nli_model_dir)
    path = tmp_path / "pipeline_test.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("VERIFACT_CONFIG", str(path))
    return path


def test_example_picker_fills_the_text_box(app_config):
    forwards = read_jsonl(ROOT / "data" / "sample" / "demo_forwards.jsonl")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    picker = at.selectbox[0]
    assert len(picker.options) == len(forwards) + 1          # "(type your own)" + every forward
    assert at.text_area[0].value == ""                       # nothing pre-filled by default
    assert at.sidebar.text_input[0].value == str(app_config)  # VERIFACT_CONFIG is picked up

    hindi = next(i for i, r in enumerate(forwards) if r["lang"] == "hi")
    picker.select_index(hindi + 1).run()
    assert at.text_area[0].value == forwards[hindi]["message"]
    assert not any("expected" in str(m.value).lower() for m in at.markdown)  # answers aren't leaked


def test_check_button_shows_a_verdict(app_config):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].select_index(1).run()
    at.button[0].click().run()
    assert not at.exception
    shown = [e.value for kind in ("error", "success", "warning", "info") for e in getattr(at, kind)]
    assert any(v in s for s in shown for v in ("FALSE", "TRUE", "PARTLY VERIFIED", "UNVERIFIED"))
    assert len(at.expander) >= 1                               # one box per extracted claim
