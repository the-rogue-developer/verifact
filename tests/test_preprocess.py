from verifact.preprocess.claims import ClaimExtractor, is_checkworthy_heuristic, split_sentences
from verifact.preprocess.clean import clean_text
from verifact.preprocess.langid import detect_language


def test_clean_removes_urls_emojis_and_forward_junk():
    msg = "🚨🚨 Forwarded as received!!! Govt banning Rs 500 notes from Dec 1 👉 https://bit.ly/x #NoteBandi"
    out = clean_text(msg)
    assert "http" not in out
    assert "🚨" not in out and "👉" not in out
    assert "forwarded" not in out.lower()
    assert "!!" not in out
    assert "Rs 500" in out and "NoteBandi" in out


def test_clean_keeps_devanagari_intact():
    msg = "सरकार 500 के नोट बंद कर रही है।।"
    out = clean_text(msg)
    assert "सरकार 500 के नोट बंद कर रही है" in out
    assert out.endswith("।") and not out.endswith("।।")


def test_langid():
    assert detect_language("The government is banning 500 rupee notes from December") == "en"
    assert detect_language("सरकार 1 दिसंबर से सभी 500 के नोट बंद कर रही है") == "hi"
    assert detect_language("Sarkar 1 December se sabhi 500 ke note band kar rahi hai, jaldi bank jao") == "hi-Latn"
    # "the", "main" and "par" alone must not trigger Hinglish
    assert detect_language("The main road in the park was closed for repairs") == "en"


def test_split_sentences_handles_danda():
    assert split_sentences("पहला वाक्य है। दूसरा वाक्य है! Third one?") == ["पहला वाक्य है।", "दूसरा वाक्य है!", "Third one?"]


def test_heuristic_filters_non_claims():
    assert not is_checkworthy_heuristic("Good morning friends, have a nice day")
    assert not is_checkworthy_heuristic("Is this really true?")
    assert not is_checkworthy_heuristic("Too short")
    assert is_checkworthy_heuristic("The RBI will ban all Rs 500 notes from 1 December")


def test_claim_extractor_falls_back_to_longest_sentence():
    ex = ClaimExtractor()
    assert ex("Good morning!") == ["Good morning!"]
    claims = ex("Good morning friends! The RBI will ban all Rs 500 notes from 1 December. Share it!")
    assert claims == ["The RBI will ban all Rs 500 notes from 1 December."]


def test_claim_extractor_with_scorer():
    ex = ClaimExtractor(scorer=lambda sents: [0.9 if "RBI" in s else 0.1 for s in sents], max_claims=2)
    assert ex("Hello there everyone. The RBI will ban notes from Monday.") == ["The RBI will ban notes from Monday."]
