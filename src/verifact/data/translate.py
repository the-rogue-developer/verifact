"""Machine translation of claim/evidence pairs (to build Hindi training data).

Default model: NLLB-200 distilled 600M (Meta), loaded with plain
AutoModelForSeq2SeqLM so it works on transformers 4.x and 5.x. Language codes
use NLLB's format: eng_Latn, hin_Deva, ben_Beng, ...

Alternatives: AI4Bharat IndicTrans2 often gives better Indic translations,
but needs the IndicTransToolkit for pre/post-processing. Comparing the two is
a nice small ablation.

For the report: translated data is "silver" data, so translation errors
become label noise. Keep a human-checked sample of ~100 rows and report how
often the label survived translation.
"""

from __future__ import annotations

from typing import Callable

TranslateFn = Callable[[list[str]], list[str]]


def nllb_translator(model_name: str = "facebook/nllb-200-distilled-600M", src: str = "eng_Latn",
                    tgt: str = "hin_Deva", device: str | None = None, batch_size: int = 16,
                    max_length: int = 400, num_beams: int = 2) -> TranslateFn:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()
    tgt_id = tok.convert_tokens_to_ids(tgt)

    @torch.inference_mode()
    def translate(texts: list[str]) -> list[str]:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(device)
            gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=max_length,
                                 num_beams=num_beams)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    return translate


def translate_pairs(rows: list[dict], translate: TranslateFn, lang: str = "hi",
                    translate_evidence: bool = True, chunk: int = 256,
                    progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Translate claims (and optionally evidence). Labels are kept.

    ``translate_evidence=False`` keeps English evidence with Hindi claims:
    the cross-lingual setting where a Hindi forward is checked against an
    English source. That's realistic and worth training on too.
    """
    out = []
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        claims = translate([r["claim"] for r in part])
        evidences = translate([r["evidence"] for r in part]) if translate_evidence else [r["evidence"] for r in part]
        for r, c, e in zip(part, claims, evidences):
            out.append({**r, "id": f"{r['id']}-mt-{lang}", "claim": c, "evidence": e,
                        "lang": lang if translate_evidence else f"{lang}-claim/en-evidence",
                        "source": f"{r.get('source', 'unk')}-mt", "orig_claim": r["claim"]})
        if progress:
            progress(min(start + chunk, len(rows)), len(rows))
    return out
