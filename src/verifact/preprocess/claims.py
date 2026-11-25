"""Claim extraction: split a message into sentences and keep the check-worthy ones.

Two modes:
    * Heuristic (default, no model needed): drops greetings, questions,
      calls to action and very short sentences.
    * Model: a fine-tuned check-worthiness classifier (train it with
      configs/claim_detector.yaml on ClaimBuster / CheckThat! data).
"""

from __future__ import annotations

import re
from typing import Callable

# Sentence boundary: . ! ? or the Devanagari danda (।), followed by space or end.
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")

_NON_CLAIM_PATTERNS = re.compile(
    r"^(good morning|good night|hello|hi|dear (friends|all)|namaste|jai hind"
    r"|सुप्रभात|नमस्ते|शुभ रात्रि|please|kripya|कृपया|share|forward)\b",
    re.IGNORECASE,
)
_HAS_CONTENT = re.compile(r"\d|[ऀ-ॿ]{2,}|[A-Z][a-z]+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def is_checkworthy_heuristic(sentence: str, min_words: int = 5) -> bool:
    s = sentence.strip()
    if len(s.split()) < min_words:
        return False
    if s.endswith("?"):
        return False
    if _NON_CLAIM_PATTERNS.match(s):
        return False
    return bool(_HAS_CONTENT.search(s))


class ClaimExtractor:
    """Pick out the check-worthy sentences of a message.

    Args:
        scorer: optional function list[str] -> list[float] giving the
            probability that each sentence is check-worthy (see
            ``verifact.models.classifier.TextClassifier``). Without one,
            the heuristic is used.
        threshold: minimum score for a sentence to count as a claim.
        max_claims: cap on claims per message, to bound latency.
    """

    def __init__(self, scorer: Callable[[list[str]], list[float]] | None = None,
                 threshold: float = 0.5, max_claims: int = 5):
        self.scorer = scorer
        self.threshold = threshold
        self.max_claims = max_claims

    def __call__(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if self.scorer is None:
            claims = [s for s in sentences if is_checkworthy_heuristic(s)]
        else:
            scores = self.scorer(sentences)
            ranked = sorted(zip(sentences, scores), key=lambda x: -x[1])
            claims = [s for s, p in ranked if p >= self.threshold]
        if not claims:
            # Short single-sentence messages: verify the whole message.
            claims = [max(sentences, key=len)]
        return claims[: self.max_claims]
