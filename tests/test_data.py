import pandas as pd
import pytest

from verifact.data.fever import _flatten_evidence
from verifact.data.tabular import load_pairs, load_style, pairs_to_style
from verifact.data.xfact import load_xfact
from verifact.io import read_jsonl, split_rows, write_jsonl
from verifact.labels import NEI, REFUTES, SUPPORTS, normalise_label

XFACT_HEADER = ["language", "site", "evidence_1", "evidence_2", "evidence_3", "evidence_4", "evidence_5",
                "link_1", "link_2", "link_3", "link_4", "link_5", "claimDate", "reviewDate", "claimant",
                "claim", "label"]


def _xfact_row(lang, claim, label, ev="snippet one"):
    ev2 = "snippet two" if ev else ""  # ev="" means a row with no evidence at all
    return [lang, "site.in", ev, ev2, "", "", "", "l1", "l2", "", "", "", "", "", "", claim, label]


def test_xfact_loader_maps_and_filters(tmp_path):
    p = tmp_path / "train.all.tsv"
    rows = [
        _xfact_row("hi", "दावा एक", "false"),
        _xfact_row("hi", "दावा दो", "mostly true"),
        _xfact_row("hi", "दावा तीन", "partly true/misleading"),
        _xfact_row("hi", "दावा चार", "other"),
        _xfact_row("hi", "दावा पांच", "true", ev=""),
        _xfact_row("en", "claim six", "true"),
    ]
    p.write_text("\n".join("\t".join(r) for r in [XFACT_HEADER] + rows), encoding="utf-8")

    out = load_xfact(p, languages=["hi"])
    assert [(r["claim"], r["label"]) for r in out] == [("दावा एक", REFUTES), ("दावा दो", SUPPORTS)]
    assert out[0]["evidence"] == "snippet one snippet two"

    amb = load_xfact(p, languages=["hi"], keep_ambiguous=True)
    assert [r["label"] for r in amb] == [REFUTES, SUPPORTS, NEI]
    assert len(load_xfact(p)) == 3  # all languages


def test_flatten_evidence_variants():
    assert _flatten_evidence("plain") == "plain"
    assert _flatten_evidence(["a", "b"]) == "a b"
    assert _flatten_evidence([["Taj_Mahal", 0, "It is in Agra."]]) == "Taj Mahal: It is in Agra."
    assert _flatten_evidence(None) == ""


def test_normalise_label():
    assert normalise_label("entailment") == SUPPORTS
    assert normalise_label("REFUTES") == REFUTES
    assert normalise_label("Not Enough Info") == NEI
    with pytest.raises(ValueError):
        normalise_label("maybe")


def test_tabular_loaders(tmp_path):
    p = tmp_path / "icb.csv"
    pd.DataFrame({"Claim": ["c1", "c2", None], "Evidence": ["e1", "e2", "e3"],
                  "Label": ["SUPPORTS", "REFUTES", "SUPPORTS"]}).to_csv(p, index=False)
    rows = load_pairs(p, "Claim", "Evidence", "Label", lang="hi", source="icb")
    assert [r["label"] for r in rows] == [SUPPORTS, REFUTES]
    with pytest.raises(KeyError):
        load_pairs(p, "claim", "Evidence", "Label")

    s = tmp_path / "style.csv"
    pd.DataFrame({"t": ["x", "y"], "l": ["Fake", "True"]}).to_csv(s, index=False)
    assert [r["label"] for r in load_style(s, "t", "l", ["fake"])] == ["FAKE", "REAL"]
    assert [r["label"] for r in pairs_to_style(rows)] == ["REAL", "FAKE"]


def test_jsonl_roundtrip_and_split(tmp_path):
    rows = [{"id": str(i), "claim": f"दावा {i}"} for i in range(20)]
    write_jsonl(rows, tmp_path / "x.jsonl")
    assert read_jsonl(tmp_path / "x.jsonl") == rows
    tr, dv, te = split_rows(rows, 0.1, 0.2)
    assert (len(tr), len(dv), len(te)) == (14, 2, 4)
    assert {r["id"] for r in tr + dv + te} == {r["id"] for r in rows}


def test_translate_pairs_with_fake_translator():
    from verifact.data.translate import translate_pairs

    rows = [{"id": "1", "claim": "a", "evidence": "b", "label": SUPPORTS, "source": "fever"},
            {"id": "2", "claim": "c", "evidence": "d", "label": REFUTES, "source": "fever"}]
    fake = lambda texts: [f"HI({t})" for t in texts]  # noqa: E731
    out = translate_pairs(rows, fake, chunk=1)
    assert [(r["claim"], r["evidence"], r["label"]) for r in out] == [
        ("HI(a)", "HI(b)", SUPPORTS), ("HI(c)", "HI(d)", REFUTES)]
    assert out[0]["orig_claim"] == "a" and out[0]["source"] == "fever-mt"
    claims_only = translate_pairs(rows, fake, translate_evidence=False)
    assert claims_only[0]["evidence"] == "b"
