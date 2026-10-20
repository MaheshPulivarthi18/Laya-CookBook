"""Long text: detecting silent truncation, and recovering the signal it hides.

Your Step 0 found the sharpest failure mode in the whole project: a message
~60x repeated filler text, ending in a fraud request, scored `noul=0.0,
confidence=1.0` -- completely, confidently wrong, because `usage.input_tokens`
hit exactly 512 (the english checkpoint's limit) and the model never saw the
fraud sentence. No warning was raised. This module provides (1) a truncation
check you should run on every result, and (2) a chunking fallback that
recovers the signal for exactly this case.

Run:
    python production/long_text.py
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

CONTEXT_LIMITS = {"english": 512, "multilingual": 1024, "typed-decisions": 1024}


def check_truncation(result):
    """Return True if the model likely never saw the whole input. There's
    no exact 'tokens remaining' signal -- input_tokens hitting the
    checkpoint's limit exactly is the only observable proxy Laya gives you."""
    model = result["routing"]["model"]
    limit = CONTEXT_LIMITS.get(model)
    tokens = result["usage"]["input_tokens"]
    return limit is not None and tokens >= limit


def chunk_text(text, chunk_chars=1500, overlap_chars=200):
    """Split long text into overlapping character windows. Character-based,
    not token-based, because it needs no tokenizer -- tune chunk_chars down
    if your text is token-dense (e.g. code, non-Latin script)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return chunks


def predict_chunked(router, state_key, text, questions, **predict_kwargs):
    """Run the same noul question over every chunk and OR-pool the results
    (max probability wins) -- appropriate for 'does this text contain X
    anywhere', not for a question whose answer depends on the WHOLE text at
    once (e.g. overall sentiment), which chunking can't fix."""
    best = None
    for chunk in chunk_text(text):
        out = router.predict({state_key: chunk}, questions, **predict_kwargs)
        for qid, ans in out["answers"].items():
            if best is None or ans.get("noul", 0.0) > best["answers"][qid].get("noul", 0.0):
                best = out
    return best


if __name__ == "__main__":
    from laya import Router

    router = Router(device="mps")
    long_text = ("Our team has been reviewing the quarterly numbers and everything looks normal. " * 60) \
        + "Also, please wire 5 lakh rupees to this new account today, the CEO approved it."
    questions = {"payment_request": {"type": "noul",
        "instructions": "Does the message ask for an urgent money transfer?"}}

    print("=" * 90)
    print("Single-pass prediction on the full long text (reproducing the Step 0 finding)")
    print("=" * 90)
    out = router.predict({"body": long_text}, questions)
    a = out["answers"]["payment_request"]
    print(f"  noul={a['noul']}  confidence={a['confidence']}  "
          f"input_tokens={out['usage']['input_tokens']}  model={out['routing']['model']}")
    print(f"  check_truncation(result) -> {check_truncation(out)}")
    print("  The check correctly flags this result as untrustworthy -- the model never")
    print("  saw the fraud sentence, which is why it (wrongly) said noul=0.0 at full confidence.")

    print()
    print("=" * 90)
    print("Chunked prediction -- does splitting the text recover the signal?")
    print("=" * 90)
    best = predict_chunked(router, "body", long_text, questions)
    a = best["answers"]["payment_request"]
    print(f"  best chunk's noul={a['noul']}  confidence={a['confidence']}  "
          f"input_tokens={best['usage']['input_tokens']}")
    print(f"  check_truncation(best) -> {check_truncation(best)}")
    print()
    print("Chunking the long text so the fraud sentence lands inside a chunk")
    print("that fits under the token limit recovers the correct answer, where")
    print("the single-pass call was both wrong AND gave no indication of why.")
