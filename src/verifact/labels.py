"""Label definitions shared by every component.

Claim verification uses the FEVER label set:
    SUPPORTS  - the evidence confirms the claim
    REFUTES   - the evidence contradicts the claim
    NOT ENOUGH INFO - the evidence neither confirms nor contradicts it

Style baselines (fake/real classifiers) use a binary label set.
"""

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NEI = "NOT ENOUGH INFO"

NLI_LABELS = [SUPPORTS, REFUTES, NEI]
NLI_LABEL2ID = {label: i for i, label in enumerate(NLI_LABELS)}
NLI_ID2LABEL = {i: label for label, i in NLI_LABEL2ID.items()}

REAL = "REAL"
FAKE = "FAKE"
STYLE_LABELS = [REAL, FAKE]
STYLE_LABEL2ID = {label: i for i, label in enumerate(STYLE_LABELS)}

# Common spellings found in public datasets, normalised to our labels.
_ALIASES = {
    "supports": SUPPORTS, "support": SUPPORTS, "supported": SUPPORTS,
    "entailment": SUPPORTS, "true": SUPPORTS,
    "refutes": REFUTES, "refute": REFUTES, "refuted": REFUTES,
    "contradiction": REFUTES, "false": REFUTES,
    "not enough info": NEI, "not_enough_info": NEI, "nei": NEI,
    "neutral": NEI, "not enough information": NEI,
    "real": REAL, "fake": FAKE,
}


def normalise_label(raw) -> str:
    """Map a dataset's label spelling onto our label names.

    Raises ValueError for labels we don't recognise, so silent data bugs
    don't creep into training.
    """
    key = str(raw).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if str(raw).strip() in NLI_LABELS + STYLE_LABELS:
        return str(raw).strip()
    raise ValueError(f"Unknown label: {raw!r}")
