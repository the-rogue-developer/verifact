"""Collect published fact-checks with the Google Fact Check Tools API, as
documents for the retrieval corpus.

The API returns ClaimReview records from IFCN-style fact-checkers (PIB Fact
Check, Alt News, BOOM, Factly, Vishvas News, ...). You need a free API key:
Google Cloud console -> enable "Fact Check Tools API" -> create an API key.

    export FACTCHECK_API_KEY=...
    python scripts/fetch_factchecks.py --queries data/queries.txt --lang hi --out data/raw/factchecks_hi.jsonl
    python scripts/fetch_factchecks.py --queries data/queries.txt --lang en --out data/raw/factchecks_en.jsonl

``queries.txt`` holds one topic per line (e.g. "currency notes", "vaccine",
"election commission", "WhatsApp forward"). Broad, common topics give the most
coverage. The output feeds ``prepare_data.py corpus --docs``.

Licence note: store the claim text, rating, publisher and URL (which the API
provides), not full scraped article text, and link to the original in the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def search(query: str, lang: str, key: str, max_pages: int = 5, page_size: int = 100) -> list[dict]:
    results, token = [], None
    for _ in range(max_pages):
        params = {"query": query, "languageCode": lang, "pageSize": page_size, "key": key}
        if token:
            params["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=30) as resp:
            data = json.load(resp)
        results += data.get("claims", [])
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return results


def claim_to_docs(c: dict, lang: str) -> list[dict]:
    docs = []
    for rv in c.get("claimReview", []):
        publisher = (rv.get("publisher") or {}).get("name", "")
        rating = rv.get("textualRating", "")
        title = rv.get("title", "")
        text = f"Claim: {c.get('text', '')}. Fact-check by {publisher}: {rating}."
        if title:
            text += f" {title}"
        docs.append({
            "id": rv.get("url") or f"{publisher}:{c.get('text', '')[:50]}",
            "title": f"Fact-check: {publisher}",
            "text": text,
            "url": rv.get("url", ""),
            "lang": rv.get("languageCode", lang),
            "source": "factcheck:" + publisher,
            "rating": rating,
            "claim_date": c.get("claimDate", ""),
        })
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--lang", default="en")
    p.add_argument("--out", required=True)
    p.add_argument("--max-pages", type=int, default=5)
    a = p.parse_args()
    key = os.environ.get("FACTCHECK_API_KEY")
    if not key:
        raise SystemExit("Set FACTCHECK_API_KEY first")

    queries = [q.strip() for q in Path(a.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    seen, out = set(), []
    for q in queries:
        try:
            claims = search(q, a.lang, key, a.max_pages)
        except Exception as e:  # keep going if one query fails
            print(f"[warn] {q!r}: {e}")
            continue
        for c in claims:
            for d in claim_to_docs(c, a.lang):
                if d["id"] not in seen:
                    seen.add(d["id"])
                    out.append(d)
        print(f"{q!r}: {len(claims)} claims, total docs {len(out)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} fact-check documents to {a.out}")


if __name__ == "__main__":
    main()
