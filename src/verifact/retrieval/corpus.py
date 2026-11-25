"""Building the evidence corpus (the knowledge base retrieval searches over).

Every passage is a dict: {"id", "title", "text", "url", "lang", "source"}.
Long documents (fact-check articles, Wikipedia pages) are split into
overlapping chunks of about ``chunk_words`` words, because retrievers and the
verifier work best on short passages.
"""

from __future__ import annotations

from ..io import read_jsonl


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 30) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []
    step = max(chunk_words - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def documents_to_passages(docs: list[dict], chunk_words: int = 120, overlap: int = 30) -> list[dict]:
    passages = []
    for d in docs:
        doc_id = str(d.get("id") or len(passages))
        for j, chunk in enumerate(chunk_text(d.get("text", ""), chunk_words, overlap)):
            passages.append({
                "id": f"{doc_id}#{j}",
                "doc_id": doc_id,
                "title": d.get("title", ""),
                "text": chunk,
                "url": d.get("url", ""),
                "lang": d.get("lang", "unk"),
                "source": d.get("source", ""),
            })
    return passages


def load_corpus(path: str) -> list[dict]:
    passages = read_jsonl(path)
    seen = set()
    for p in passages:
        if "id" not in p or "text" not in p:
            raise ValueError("Every corpus row needs 'id' and 'text'")
        if p["id"] in seen:
            raise ValueError(f"Duplicate passage id {p['id']!r}")
        seen.add(p["id"])
    return passages


def passage_text(p: dict) -> str:
    """Text used for indexing and verification: title plus body."""
    title = (p.get("title") or "").strip()
    return f"{title}. {p['text']}" if title else p["text"]
