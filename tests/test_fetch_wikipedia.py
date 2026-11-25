"""scripts/fetch_wikipedia.py with mocked HTTP (offline): request parameters,
User-Agent, batching, redirects, missing pages, continuation, retry, categories."""

import importlib.util
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from verifact.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fw():
    spec = importlib.util.spec_from_file_location("fetch_wikipedia", ROOT / "scripts" / "fetch_wikipedia.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeWiki:
    """Stands in for urllib.request.urlopen and answers like the MediaWiki API."""

    ARTICLES = {
        "Ranchi": "Ranchi is the capital of the Indian state of Jharkhand.\n\nIt is known for waterfalls.",
        "Taj Mahal": "The Taj Mahal is a mausoleum in Agra.",
        "AT&T": "AT&T is a telecommunications company.",
    }
    REDIRECTS = {"Ranchi city": "Ranchi"}

    def __init__(self, fail_first: int = 0):
        self.calls = []          # (params, headers)
        self.fail_first = fail_first

    def urlopen(self, req, timeout=None):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(req.full_url).query))
        self.calls.append((params, dict(req.header_items())))
        if self.fail_first:
            self.fail_first -= 1
            raise urllib.error.URLError("connection reset")
        if params.get("list") == "categorymembers":
            body = {"query": {"categorymembers": [{"ns": 0, "title": "Taj Mahal"}, {"ns": 0, "title": "Ranchi"}]}}
        else:
            pages, redirects = [], []
            for t in params["titles"].split("|"):
                target = self.REDIRECTS.get(t, t)
                if target != t:
                    redirects.append({"from": t, "to": target})
                if target in self.ARTICLES:
                    pages.append({"pageid": len(pages) + 1, "ns": 0, "title": target, "extract": self.ARTICLES[target]})
                else:
                    pages.append({"ns": 0, "title": t, "missing": True})
            body = {"batchcomplete": True, "query": {"redirects": redirects, "pages": pages}}
        return io.BytesIO(json.dumps(body).encode("utf-8"))


@pytest.fixture
def fake(monkeypatch, fw):
    f = FakeWiki()
    monkeypatch.setattr(fw.urllib.request, "urlopen", f.urlopen)
    monkeypatch.setattr(fw.time, "sleep", lambda s: f.calls.append(("sleep", s)))
    return f


def test_fetch_titles_end_to_end(fw, fake, tmp_path):
    titles = tmp_path / "titles.txt"
    titles.write_text("# comment\nRanchi city\n\nRanchi\nTaj Mahal\nNo Such Article\nAT&T\n", encoding="utf-8")
    out = tmp_path / "wiki.jsonl"
    assert fw.main(["--titles", str(titles), "--lang", "en", "--out", str(out)]) == 3

    params, headers = fake.calls[0]
    assert params["action"] == "query" and params["prop"] == "extracts"
    assert params["exintro"] == "1" and params["explaintext"] == "1" and params["redirects"] == "1"
    assert params["titles"] == "Ranchi city|Ranchi|Taj Mahal|No Such Article|AT&T"  # one batch
    assert headers["User-agent"] == fw.DEFAULT_USER_AGENT

    docs = {d["id"]: d for d in read_jsonl(out)}
    # redirect + direct title give one doc; the missing page is skipped
    assert set(docs) == {"wiki-en-Ranchi", "wiki-en-Taj_Mahal", "wiki-en-AT&T"}
    ranchi = docs["wiki-en-Ranchi"]
    assert ranchi == {"id": "wiki-en-Ranchi", "title": "Ranchi",
                      "text": "Ranchi is the capital of the Indian state of Jharkhand. It is known for waterfalls.",
                      "url": "https://en.wikipedia.org/wiki/Ranchi", "lang": "en", "source": "wikipedia"}
    assert docs["wiki-en-Taj_Mahal"]["url"] == "https://en.wikipedia.org/wiki/Taj_Mahal"
    assert docs["wiki-en-AT&T"]["url"] == "https://en.wikipedia.org/wiki/AT%26T"


def test_batches_of_20_with_pause_between_requests(fw, fake):
    client = fw.WikiClient("en")
    docs, failed = fw.fetch(client, [f"Title {i}" for i in range(45)])
    requests = [c for c in fake.calls if c[0] != "sleep"]
    assert [len(p["titles"].split("|")) for p, _ in requests] == [20, 20, 5]
    assert [c for c in fake.calls if c[0] == "sleep"] == [("sleep", 0.5)] * 2
    assert docs == [] and failed == []


def test_retry_once_then_skip_batch(fw, monkeypatch):
    sleeps = []
    ok = FakeWiki(fail_first=1)
    monkeypatch.setattr(fw.urllib.request, "urlopen", ok.urlopen)
    docs, failed = fw.fetch(fw.WikiClient("en", sleep=sleeps.append), ["Ranchi"])
    assert [d["title"] for d in docs] == ["Ranchi"] and failed == [] and len(ok.calls) == 2

    bad = FakeWiki(fail_first=2)
    monkeypatch.setattr(fw.urllib.request, "urlopen", bad.urlopen)
    docs, failed = fw.fetch(fw.WikiClient("en", sleep=sleeps.append), ["Ranchi", "Taj Mahal"])
    assert docs == [] and failed == ["Ranchi", "Taj Mahal"] and len(bad.calls) == 2  # exactly one retry


def test_continuation_is_followed(fw):
    responses = iter([
        {"continue": {"excontinue": 1, "continue": "||"},
         "query": {"pages": [{"title": "A", "extract": "Lead of A."}, {"title": "B"}]}},
        {"query": {"pages": [{"title": "A"}, {"title": "B", "extract": "Lead of B."}]}},
    ])
    seen = []
    client = fw.WikiClient("en", http_get=lambda url, params, headers: (seen.append(params), next(responses))[1],
                           sleep=lambda s: None)
    pages = client.extracts(["A", "B"])
    assert {p["title"]: p["extract"] for p in pages} == {"A": "Lead of A.", "B": "Lead of B."}
    assert seen[1]["excontinue"] == 1


def test_categories_and_hindi_urls(fw, fake, tmp_path):
    out = tmp_path / "cat.jsonl"
    fw.main(["--categories", "Prime Ministers of India", "--lang", "hi", "--out", str(out)])
    cat_params = fake.calls[0][0]
    assert cat_params["list"] == "categorymembers" and cat_params["cmtitle"] == "Category:Prime Ministers of India"
    assert cat_params["cmlimit"] == "100" and cat_params["cmnamespace"] == "0"
    assert {d["id"] for d in read_jsonl(out)} == {"wiki-hi-Taj_Mahal", "wiki-hi-Ranchi"}
    # Devanagari stays readable in URLs; spaces become underscores
    assert fw.article_url("भारत के प्रधानमंत्री", "hi") == "https://hi.wikipedia.org/wiki/भारत_के_प्रधानमंत्री"


def test_topic_files_are_clean():
    for name, minimum in (("wiki_topics_en.txt", 250), ("wiki_topics_hi.txt", 50)):
        path = ROOT / "data" / name
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        titles = [t for t in lines if t and not t.startswith("#")]
        assert len(titles) >= minimum and len(titles) == len(set(titles)), name
        assert all("|" not in t and "_" not in t for t in titles), name  # "|" would break batching
