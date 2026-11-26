"""Build a retrieval corpus from Wikipedia lead sections (MediaWiki API, no key).

    python scripts/fetch_wikipedia.py --titles data/wiki_topics_en.txt --lang en \
        --out data/raw/wiki_en.jsonl
    python scripts/fetch_wikipedia.py --titles data/wiki_topics_hi.txt --lang hi \
        --out data/raw/wiki_hi.jsonl
    python scripts/fetch_wikipedia.py --lang en --categories "Prime Ministers of India" \
        --out data/raw/wiki_pm.jsonl

Then add the files to the corpus:
    python scripts/prepare_data.py corpus --docs data/raw/wiki_en.jsonl data/raw/wiki_hi.jsonl ...

Why lead sections: the intro paragraph of an article states the key facts
(who, what, when, where) in a few sentences, which is what claim
verification needs, and it keeps the corpus small enough to index on a laptop.

Etiquette (https://meta.wikimedia.org/wiki/User-Agent_policy): a descriptive
User-Agent is required, requests are sent one at a time with a pause between
them, and titles are batched (20 per request, the API's limit for intro extracts).
The policy also asks for contact details in the User-Agent: add yours with
--user-agent "VeriFact-student-project/0.1 (BIT Mesra minor project; <your email or URL>)".

Titles file: one title per line; blank lines and lines starting with # are ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifact.io import write_jsonl  # noqa: E402

log = logging.getLogger("fetch_wikipedia")

DEFAULT_USER_AGENT = "VeriFact-student-project/0.1 (BIT Mesra minor project)"
BATCH = 20                  # TextExtracts returns at most 20 intro extracts per request
CATEGORY_LIMIT = 100
SLEEP = 0.5
RETRY_WAIT = 2.0
# HTTP 429 (Too Many Requests) / 503 mean "slow down", not "broken": a 2 s retry
# is far too short (in Colab this lost 70 articles). Wait as long as the
# server's Retry-After header says, otherwise use this growing schedule.
RATE_LIMIT_CODES = (429, 503)
RATE_LIMIT_WAITS = (5.0, 15.0, 45.0, 90.0)
MAX_WAIT = 120.0
# ASCII characters that are safe unescaped in the /wiki/ path; everything else
# ASCII (?, #, &, %, ", ...) is percent-encoded. Non-ASCII (Devanagari) is kept
# as is, so Hindi links stay readable (browsers accept these IRIs).
_URL_SAFE_ASCII = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.,:;()!*'~/@$+=")


class WikiError(RuntimeError):
    pass


def article_url(title: str, lang: str) -> str:
    path = "".join(c if ord(c) > 127 or c in _URL_SAFE_ASCII else urllib.parse.quote(c)
                   for c in title.replace(" ", "_"))
    return f"https://{lang}.wikipedia.org/wiki/{path}"


def read_titles(path: str | Path) -> list[str]:
    titles = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    return list(dict.fromkeys(titles))  # drop duplicates, keep order


def urllib_get(url: str, params: dict, headers: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def retry_after(err: urllib.error.HTTPError) -> float | None:
    """Seconds from a Retry-After header, or None (missing, or the rarer HTTP-date form)."""
    value = err.headers.get("Retry-After") if err.headers else None
    try:
        return min(float(value), MAX_WAIT) if value is not None else None
    except ValueError:
        return None


class WikiClient:
    """Polite MediaWiki API client: one request at a time, a pause between
    requests, backing off when rate limited, and one retry on other errors.
    ``http_get`` is injectable for tests."""

    def __init__(self, lang: str, user_agent: str = DEFAULT_USER_AGENT,
                 http_get: Callable[[str, dict, dict], dict] = urllib_get,
                 sleep: Callable[[float], None] | None = None, pause: float = SLEEP):
        self.lang = lang
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": user_agent}
        self.http_get = http_get
        self.sleep = sleep or time.sleep
        self.pause = pause
        self._requests = 0

    def request(self, params: dict) -> dict:
        params = {"format": "json", "formatversion": "2", **params}
        errors = waits = 0
        while True:
            if self._requests:
                self.sleep(self.pause)
            self._requests += 1
            try:
                data = self.http_get(self.api, params, self.headers)
                if "error" in data:
                    raise WikiError(f"{data['error'].get('code')}: {data['error'].get('info')}")
                return data
            except urllib.error.HTTPError as e:  # before URLError: HTTPError is a subclass
                if e.code not in RATE_LIMIT_CODES:
                    err = e
                elif waits < len(RATE_LIMIT_WAITS):
                    wait = retry_after(e) or RATE_LIMIT_WAITS[waits]
                    waits += 1
                    log.warning("HTTP %d (rate limited), waiting %.0fs before retry %d/%d",
                                e.code, wait, waits, len(RATE_LIMIT_WAITS))
                    self.sleep(wait)
                    continue
                else:
                    raise WikiError(f"still rate limited after {waits} waits: {e}") from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, WikiError) as e:
                err = e
            errors += 1
            if errors == 2:
                raise WikiError(f"request failed twice: {err}") from err
            log.warning("request failed (%s), retrying once in %.0fs", err, RETRY_WAIT)
            self.sleep(RETRY_WAIT)

    def extracts(self, titles: list[str]) -> list[dict]:
        """Lead sections for up to BATCH titles. Follows redirects; skips missing pages."""
        params = {"action": "query", "prop": "extracts", "exintro": "1", "explaintext": "1",
                  "redirects": "1", "exlimit": str(BATCH), "titles": "|".join(titles)}
        pages: dict[str, dict] = {}
        cont: dict = {}
        while True:  # long extracts can be split over several responses ("continue")
            data = self.request({**params, **cont})
            for p in data.get("query", {}).get("pages", []):
                if p.get("missing") or p.get("invalid"):
                    log.info("skip missing page: %s", p.get("title"))
                    continue
                page = pages.setdefault(p["title"], {"title": p["title"], "extract": ""})
                if p.get("extract"):
                    page["extract"] = p["extract"]
            if "continue" not in data:
                break
            cont = data["continue"]
        return list(pages.values())

    def category_members(self, category: str, limit: int = CATEGORY_LIMIT) -> list[str]:
        """Article titles in a category (namespace 0 only, no subcategories)."""
        name = category if ":" in category else f"Category:{category}"
        data = self.request({"action": "query", "list": "categorymembers", "cmtitle": name,
                             "cmnamespace": "0", "cmtype": "page", "cmlimit": str(min(limit, 500))})
        return [m["title"] for m in data.get("query", {}).get("categorymembers", [])][:limit]


def to_doc(page: dict, lang: str) -> dict:
    title = page["title"]
    return {
        "id": f"wiki-{lang}-{title.replace(' ', '_')}",
        "title": title,
        "text": " ".join(page["extract"].split()),  # collapse the API's newlines
        "url": article_url(title, lang),
        "lang": lang,
        "source": "wikipedia",
    }


def fetch(client: WikiClient, titles: list[str], categories: list[str] | None = None,
          category_limit: int = CATEGORY_LIMIT) -> tuple[list[dict], list[str]]:
    """Return (docs, titles of batches that failed even after the retry)."""
    titles = list(titles)
    for cat in categories or []:
        try:
            members = client.category_members(cat, category_limit)
        except WikiError as e:
            log.warning("category %r failed: %s", cat, e)
            continue
        log.info("category %r: %d articles", cat, len(members))
        titles += members
    titles = list(dict.fromkeys(titles))

    docs: dict[str, dict] = {}
    failed: list[str] = []
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i + BATCH]
        try:
            pages = client.extracts(batch)
        except WikiError as e:
            log.warning("batch %d failed, skipping %d titles: %s", i // BATCH + 1, len(batch), e)
            failed += batch
            continue
        for p in pages:
            if not p["extract"].strip():
                log.info("skip page with empty lead section: %s", p["title"])
                continue
            doc = to_doc(p, client.lang)
            docs.setdefault(doc["id"], doc)  # two titles can redirect to one article
        log.info("%d/%d titles requested, %d docs", min(i + BATCH, len(titles)), len(titles), len(docs))
    return list(docs.values()), failed


def main(argv: list[str] | None = None) -> int:
    """Exit status: 0 if every batch was fetched, 1 if some titles failed (the
    file is still written), so a notebook can avoid caching an incomplete corpus."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--titles", help="file with one article title per line")
    p.add_argument("--categories", nargs="*", default=[],
                   help='category names, e.g. "Prime Ministers of India" (member articles are added)')
    p.add_argument("--category-limit", type=int, default=CATEGORY_LIMIT)
    p.add_argument("--lang", choices=["en", "hi"], default="en")
    p.add_argument("--out", required=True)
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    a = p.parse_args(argv)
    if not a.titles and not a.categories:
        p.error("pass --titles and/or --categories")

    titles = read_titles(a.titles) if a.titles else []
    docs, failed = fetch(WikiClient(a.lang, a.user_agent), titles, a.categories, a.category_limit)
    n = write_jsonl(docs, a.out)
    log.info("wrote %d documents to %s", n, a.out)
    if failed:
        log.warning("%d titles not fetched because of errors (re-run to retry): %s",
                    len(failed), ", ".join(failed[:10]) + (" ..." if len(failed) > 10 else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
