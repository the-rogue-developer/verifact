from verifact.eval.metrics import classification_metrics, fever_style_score, mrr, recall_at_k
from verifact.labels import NEI, REFUTES, SUPPORTS
from verifact.pipeline.aggregate import EvidenceJudgement, aggregate_claim, aggregate_message


def _j(s, r, n, pid="p"):
    return EvidenceJudgement({"id": pid, "text": pid}, {SUPPORTS: s, REFUTES: r, NEI: n})


def test_claim_refuted_by_top_evidence():
    v = aggregate_claim("c", [_j(0.05, 0.9, 0.05, "p1"), _j(0.7, 0.1, 0.2, "p2")])
    assert v.label == REFUTES and v.best_evidence["id"] == "p1"


def test_rank_weighting_discounts_lower_evidence():
    # Rank 3 refutation at 0.9 * 0.49 = 0.44 < 0.6 threshold, so it doesn't count.
    v = aggregate_claim("c", [_j(0.1, 0.1, 0.8), _j(0.1, 0.1, 0.8), _j(0.05, 0.9, 0.05)])
    assert v.label == NEI


def test_claim_supported_and_nei():
    assert aggregate_claim("c", [_j(0.85, 0.05, 0.1)]).label == SUPPORTS
    assert aggregate_claim("c", [_j(0.4, 0.3, 0.3)]).label == NEI
    assert aggregate_claim("c", []).label == NEI


def test_message_verdicts():
    sup = aggregate_claim("a", [_j(0.9, 0.05, 0.05)])
    ref = aggregate_claim("b", [_j(0.05, 0.9, 0.05)])
    nei = aggregate_claim("c", [_j(0.3, 0.3, 0.4)])
    assert aggregate_message([sup, ref], "en").verdict == "FALSE"
    assert aggregate_message([sup], "en").verdict == "TRUE"
    assert aggregate_message([sup, nei], "en").verdict == "PARTLY VERIFIED"
    unv = aggregate_message([nei], "hi", style_fake_prob=0.9)
    assert unv.verdict == "UNVERIFIED" and "caution" in unv.note
    assert aggregate_message([], "en").verdict == "UNVERIFIED"
    assert isinstance(aggregate_message([ref], "en").to_dict()["claims"][0]["evidence"], list)


def test_metrics():
    m = classification_metrics([SUPPORTS, REFUTES, REFUTES, NEI], [SUPPORTS, REFUTES, SUPPORTS, NEI])
    assert m["accuracy"] == 0.75 and 0 < m["macro_f1"] < 1
    gold = [{"a"}, {"b"}, {"c"}]
    ret = [["a", "x"], ["x", "b"], ["x", "y"]]
    assert recall_at_k(gold, ret, 1) == 1 / 3
    assert recall_at_k(gold, ret, 2) == 2 / 3
    assert abs(mrr(gold, ret) - 0.5) < 1e-9
    assert fever_style_score([SUPPORTS, REFUTES, NEI], [SUPPORTS, REFUTES, NEI],
                             [{"a"}, {"z"}, set()], [["a"], ["b"], []]) == 2 / 3
