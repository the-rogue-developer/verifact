"""Text cleaning for WhatsApp-style forwards and scraped articles."""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\S+)")  # \S not \w: \w would split Hindi words at vowel signs
_REPEAT_PUNCT = re.compile(r"([!?.।])\1+")
_WHITESPACE = re.compile(r"\s+")
# Boilerplate common in forwards, in English, Hindi and Hinglish.
_FORWARD_JUNK = re.compile(
    r"(forwarded( as received)?|forward(ed)? many times|share (it|this) with everyone"
    r"|please share|must share|आगे भेजें|सभी को भेजें|शेयर करें|jaldi share karo"
    r"|sabko bhejo|copy paste)",
    re.IGNORECASE,
)


def _is_emoji_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat in ("So", "Sk", "Cs") or 0x1F000 <= ord(ch) <= 0x1FAFF or ch in "‍️"


def clean_text(text: str, keep_urls: bool = False) -> str:
    """Normalise a message before claim extraction.

    Keeps Devanagari intact (including matras and the danda ।), removes emojis,
    URLs, @mentions and forward boilerplate, and collapses repeated punctuation.
    """
    text = unicodedata.normalize("NFC", text)
    if not keep_urls:
        text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)
    text = _FORWARD_JUNK.sub(" ", text)
    text = "".join(" " if _is_emoji_or_symbol(c) else c for c in text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()
