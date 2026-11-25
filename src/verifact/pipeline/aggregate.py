"""Turning per-evidence NLI probabilities into claim and message verdicts.

Claim level: each retrieved passage gives P(SUPPORTS), P(REFUTES), P(NEI).
Each passage's vote is weighted by its retrieval rank (the top passage
counts most). The claim is REFUTED or SUPPORTED if the strongest weighted
vote clears a confidence threshold, otherwise NOT ENOUGH INFO. Refutation
is checked first: one solid contradicting source is enough to flag a claim.

Message level:
    any claim REFUTED                -> FALSE
    all claims SUPPORTED             -> TRUE
    some SUPPORTED, rest NEI         -> PARTLY VERIFIED
    all NEI                          -> UNVERIFIED
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..labels import NEI, REFUTES, SUPPORTS


@dataclass
class EvidenceJudgement:
    passage: dict
    probs: dict[str, float]


@dataclass
class ClaimVerdict:
    claim: str
    label: str
    confidence: float
    evidence: list[EvidenceJudgement] = field(default_factory=list)
    best_evidence: dict | None = None


@dataclass
class MessageVerdict:
    verdict: str            # TRUE / FALSE / PARTLY VERIFIED / UNVERIFIED
    confidence: float
    language: str
    claims: list[ClaimVerdict]
    style_fake_prob: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def rank_weights(n: int, decay: float = 0.7) -> list[float]:
    """Weight for rank 1, 2, 3...: 1, 0.7, 0.49... (relative to the top passage)."""
    return [decay ** i for i in range(n)]


def aggregate_claim(claim: str, judgements: list[EvidenceJudgement],
                    threshold: float = 0.6, decay: float = 0.7) -> ClaimVerdict:
    if not judgements:
        return ClaimVerdict(claim, NEI, 0.0, [], None)
    weights = rank_weights(len(judgements), decay)
    best = {SUPPORTS: (0.0, None), REFUTES: (0.0, None)}
    for w, j in zip(weights, judgements):
        for lab in (SUPPORTS, REFUTES):
            score = w * j.probs.get(lab, 0.0)
            if score > best[lab][0]:
                best[lab] = (score, j)
    for lab in (REFUTES, SUPPORTS):
        score, j = best[lab]
        if score >= threshold:
            return ClaimVerdict(claim, lab, round(j.probs[lab], 4), judgements, j.passage)
    top = judgements[0]
    return ClaimVerdict(claim, NEI, round(top.probs.get(NEI, 0.0), 4), judgements, top.passage)


def aggregate_message(claims: list[ClaimVerdict], language: str,
                      style_fake_prob: float | None = None,
                      style_threshold: float = 0.7) -> MessageVerdict:
    labels = [c.label for c in claims]
    if not claims:
        return MessageVerdict("UNVERIFIED", 0.0, language, [], style_fake_prob,
                              "No check-worthy claims found.")
    if REFUTES in labels:
        conf = max(c.confidence for c in claims if c.label == REFUTES)
        return MessageVerdict("FALSE", conf, language, claims, style_fake_prob)
    if all(lab == SUPPORTS for lab in labels):
        conf = min(c.confidence for c in claims)
        return MessageVerdict("TRUE", conf, language, claims, style_fake_prob)
    if SUPPORTS in labels:
        conf = max(c.confidence for c in claims if c.label == SUPPORTS)
        return MessageVerdict("PARTLY VERIFIED", conf, language, claims, style_fake_prob,
                              "Some claims are supported; no evidence was found for the others.")
    note = "No supporting or refuting evidence found in the knowledge base."
    if style_fake_prob is not None and style_fake_prob >= style_threshold:
        note += f" The writing style looks like typical misinformation ({style_fake_prob:.0%}); treat with caution."
    return MessageVerdict("UNVERIFIED", 0.0, language, claims, style_fake_prob, note)
