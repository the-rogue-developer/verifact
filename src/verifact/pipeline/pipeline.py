"""End-to-end VeriFact pipeline.

    message -> clean -> detect language -> extract claims
            -> (for each claim) retrieve evidence -> NLI verify each passage
            -> aggregate per claim -> aggregate per message (+ style fallback)

Every component is injected, so you can swap models or run ablations
(BM25-only, no re-ranker, ...) without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..preprocess.claims import ClaimExtractor
from ..preprocess.clean import clean_text
from ..preprocess.langid import detect_language
from ..retrieval.corpus import passage_text
from .aggregate import EvidenceJudgement, MessageVerdict, aggregate_claim, aggregate_message


class SearchFn(Protocol):
    def search(self, query: str, k: int = 5) -> list[dict]: ...


class VerifierFn(Protocol):
    def predict(self, claims: list[str], evidences: list[str]) -> list[dict[str, float]]: ...


@dataclass
class PipelineSettings:
    top_k_evidence: int = 3
    claim_threshold: float = 0.6
    rank_decay: float = 0.7
    style_threshold: float = 0.7


class VeriFactPipeline:
    def __init__(self, retriever: SearchFn, verifier: VerifierFn,
                 claim_extractor: ClaimExtractor | None = None,
                 style_scorer: Callable[[str], float] | None = None,
                 translator: Callable[[str, str], str] | None = None,
                 settings: PipelineSettings | None = None):
        """
        Args:
            retriever: anything with .search(query, k) -> list of passage dicts
                (HybridRetriever, or a single retriever wrapped to return passages).
            verifier: anything with .predict(claims, evidences) -> list of
                {label: prob} dicts (verifact.models.classifier.Verifier).
            claim_extractor: defaults to the heuristic extractor.
            style_scorer: optional text -> P(fake) for the no-evidence fallback.
            translator: optional (text, lang) -> English text. Leave it out to use
                the multilingual models directly; add it for the
                translate-to-English ablation.
        """
        self.retriever = retriever
        self.verifier = verifier
        self.extract_claims = claim_extractor or ClaimExtractor()
        self.style_scorer = style_scorer
        self.translator = translator
        self.s = settings or PipelineSettings()

    def verify(self, message: str) -> MessageVerdict:
        text = clean_text(message)
        lang = detect_language(text)
        if self.translator is not None and lang != "en":
            text = self.translator(text, lang)
        claims = self.extract_claims(text)

        claim_verdicts = []
        for claim in claims:
            passages = self.retriever.search(claim, k=self.s.top_k_evidence)
            probs = self.verifier.predict([claim] * len(passages),
                                          [passage_text(p) for p in passages])
            judgements = [EvidenceJudgement(p, pr) for p, pr in zip(passages, probs)]
            claim_verdicts.append(aggregate_claim(claim, judgements, self.s.claim_threshold,
                                                  self.s.rank_decay))

        style_prob = self.style_scorer(text) if self.style_scorer else None
        return aggregate_message(claim_verdicts, lang, style_prob, self.s.style_threshold)
