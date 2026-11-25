"""VeriFact demo UI.

    streamlit run app/streamlit_app.py

On Colab, expose it with a tunnel (see the notebook), or deploy to
Hugging Face Spaces.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifact.config import load_config  # noqa: E402
from verifact.pipeline.build import build_pipeline  # noqa: E402

VERDICT_STYLE = {
    "FALSE": ("❌", "error"),
    "TRUE": ("✅", "success"),
    "PARTLY VERIFIED": ("⚠️", "warning"),
    "UNVERIFIED": ("❔", "info"),
}
CLAIM_BADGE = {"REFUTES": "🔴 Refuted", "SUPPORTS": "🟢 Supported", "NOT ENOUGH INFO": "⚪ Not enough info"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "hi-Latn": "Hinglish"}


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline(config_path: str):
    return build_pipeline(load_config(config_path))


st.set_page_config(page_title="VeriFact", page_icon="🔎", layout="centered")
st.title("🔎 VeriFact")
st.caption("Evidence-based fact-checking for English, Hindi and Hinglish messages")

config_path = st.sidebar.text_input("Pipeline config", str(ROOT / "configs" / "pipeline.yaml"))
show_all = st.sidebar.checkbox("Show all retrieved evidence", value=False)

message = st.text_area("Paste a message, forward or headline", height=150,
                       placeholder="e.g. सरकार 1 दिसंबर से सभी 500 के नोट बंद कर रही है")

if st.button("Check", type="primary") and message.strip():
    with st.spinner("Extracting claims, searching evidence, verifying..."):
        result = get_pipeline(config_path).verify(message)

    icon, kind = VERDICT_STYLE.get(result.verdict, ("", "info"))
    conf = f" ({result.confidence:.0%} confidence)" if result.confidence else ""
    getattr(st, kind)(f"{icon} **{result.verdict}**{conf}")
    st.caption(f"Detected language: {LANG_NAMES.get(result.language, result.language)}")
    if result.note:
        st.write(result.note)

    for i, c in enumerate(result.claims, 1):
        with st.expander(f"Claim {i}: {c.claim}", expanded=True):
            st.markdown(f"**{CLAIM_BADGE.get(c.label, c.label)}** · {c.confidence:.0%}")
            if c.best_evidence:
                st.markdown("**Key evidence**")
                st.write(c.best_evidence["text"])
                if c.best_evidence.get("url"):
                    st.markdown(f"[Source]({c.best_evidence['url']})")
            if show_all:
                for j in c.evidence:
                    probs = ", ".join(f"{k}: {v:.2f}" for k, v in j.probs.items())
                    st.markdown(f"- {j.passage['text'][:300]}  \n  `{probs}`")

st.sidebar.markdown("---")
st.sidebar.caption("Verdicts are model predictions, not professional fact-checks. "
                   "Always check the linked sources.")
