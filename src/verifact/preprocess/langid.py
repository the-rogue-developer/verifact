"""Lightweight language identification for English / Hindi / Hinglish.

Returns one of:
    "hi"       Hindi in Devanagari script
    "hi-Latn"  Hinglish: Hindi (or code-mixed) written in Roman script
    "en"       English

Script detection is reliable. Telling Hinglish from English is the hard
part: we count distinctive romanised Hindi function words. A fastText
lid.176 model would be better for other languages, but it can't detect
romanised Hindi either. Evaluating this heuristic on a labelled sample is a
good small experiment for your report.
"""

from __future__ import annotations

import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z]+")

# High-frequency romanised Hindi words that are rare or absent in English.
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "nhi", "kya", "kyu", "kyun", "kyunki", "aur",
    "bhi", "ki", "ka", "ke", "ko", "se", "mein", "mai", "main", "hum", "tum", "aap",
    "yeh", "ye", "woh", "wo", "raha", "rahi", "rahe", "tha", "thi", "the", "kar",
    "karo", "karna", "karne", "kiya", "gaya", "gayi", "hoga", "hogi", "sab", "sabhi",
    "abhi", "jaldi", "lekin", "agar", "toh", "bahut", "kuch", "sirf", "wala", "wali",
    "sarkar", "logon", "log", "dekho", "bhai", "ji", "par", "pe", "liye", "baad",
}
# "the", "main", "par", "log" are also English, so they only count alongside other markers.
_AMBIGUOUS = {"the", "main", "par", "log", "ji", "pe"}


def detect_language(text: str, hinglish_threshold: float = 0.15) -> str:
    deva = len(_DEVANAGARI.findall(text))
    latin = len(_LATIN.findall(text))
    if deva == 0 and latin == 0:
        return "en"
    if deva >= latin * 0.5:
        return "hi"
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return "en"
    strong = sum(1 for w in words if w in HINGLISH_MARKERS and w not in _AMBIGUOUS)
    weak = sum(1 for w in words if w in _AMBIGUOUS)
    score = (strong + (weak if strong >= 2 else 0)) / len(words)
    return "hi-Latn" if strong >= 2 and score >= hinglish_threshold else "en"
