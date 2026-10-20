"""Many options: what actually happens with 130 choice labels, and which
of three fixes is safest.

00_basics/06_hidden_features.py found the default shortlist embedder can
silently drop the correct label. This module checks the other two options:
raising `head_max_len`/`max_len` to fit everything, and a manual two-step
coarse-then-fine choice -- and is honest about what each one costs.

Run:
    python production/many_options.py
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

STATE = {"body": "We were billed twice for March, please refund the duplicate "
                  "today or we will cancel our plan."}


def make_many_options(n=130):
    criteria = {f"dept_{i}": f"category number {i} for internal routing of enterprise "
                              f"support requests needing specialized handling" for i in range(n)}
    criteria["billing"] = "invoices, payments, refunds"
    return {"department": {"type": "choice",
        "instructions": "Which department should handle this request?", "criteria": criteria}}


if __name__ == "__main__":
    from laya import Router, embed_fn_from_agent, predict_shortlist

    router = Router(device="mps")
    many_q = make_many_options(130)

    print("=" * 90)
    print("1. The failure, reproduced: 130 options with default settings")
    print("=" * 90)
    try:
        router.predict(STATE, many_q)
        print("  (no error -- if you see this, the breaking point has shifted; re-check)")
    except ValueError as e:
        print(f"  ValueError: {e}")
        print("  Default head_max_len=192 can't fit 130 option markers. This ISN'T a")
        print("  silent wrong answer like truncation elsewhere in Laya -- it's a hard error,")
        print("  which is actually the safer failure mode: you find out immediately.")

    print()
    print("=" * 90)
    print("2. Fix attempt: raise head_max_len alone")
    print("=" * 90)
    try:
        router.predict(STATE, many_q, head_max_len=2048)
        print("  worked (unexpected -- re-verify)")
    except ValueError as e:
        print(f"  STILL FAILS: {e}")
        print("  head_max_len only controls how much of the token budget the HEAD")
        print("  (instructions+options) gets within the overall sequence. The final")
        print("  sequence is still capped at `max_len` (512 for english), so if head+")
        print("  options alone would exceed that, you have to raise max_len too.")

    print()
    print("=" * 90)
    print("3. Fix that actually works: raise BOTH head_max_len and max_len")
    print("=" * 90)
    out = router.predict(STATE, many_q, head_max_len=900, max_len=1024)
    a = out["answers"]["department"]
    print(f"  choice={a['choice']}  top_probability={a['probabilities'][a['choice']]:.4f}  "
          f"input_tokens={out['usage']['input_tokens']}")
    print(f"  Correct here -- but {out['usage']['input_tokens']} tokens is well past the")
    print("  english checkpoint's TRAINED context (512 tokens, per laya.router's own")
    print("  benchmark notes). You're running the model outside the range it was evaluated")
    print("  on. It worked on this one example; don't assume it generalizes untested.")

    print()
    print("=" * 90)
    print("4. Alternative: shortlist_choice (00_basics/06 already found this unreliable)")
    print("=" * 90)
    agent = router.load("english")
    embed_fn = embed_fn_from_agent(agent)
    result = predict_shortlist(agent, STATE, many_q, embed_fn, k=5)
    labels = result["shortlist"]["department"]["labels"]
    print(f"  shortlisted to k=5: {labels}")
    print(f"  'billing' in shortlist: {'billing' in labels}")
    print("  Cheap (no context-limit risk) but only as good as the fallback embedder --")
    print("  verify it keeps the right label on YOUR option set before trusting it.")

    print()
    print("=" * 90)
    print("5. Safest default: a manual two-step choice (coarse group, then fine label)")
    print("=" * 90)
    groups = {
        "billing_and_payments": "billing, payments, refunds, invoices",
        "operations": "all the internal dept_* routing categories (bugs, requests, misc ops)",
    }
    coarse_q = {"group": {"type": "choice",
        "instructions": "Which broad group should handle this request?", "criteria": groups}}
    coarse_out = router.predict(STATE, coarse_q)
    chosen_group = coarse_out["answers"]["group"]["choice"]
    print(f"  coarse choice -> {chosen_group} "
          f"(confidence {coarse_out['answers']['group']['confidence']:.3f})")
    if chosen_group == "billing_and_payments":
        fine_q = {"department": {"type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {"billing": "invoices, payments, refunds"}}}
    else:
        fine_q = many_q
    fine_out = router.predict(STATE, fine_q)
    print(f"  fine choice -> {fine_out['answers']['department']['choice']}")
    print("  This costs a second forward pass and requires YOU to design the grouping,")
    print("  but it needs no undocumented context-limit stretching and no embedding")
    print("  similarity step to trust. For a real deployment with 100+ genuinely similar")
    print("  options, this is the approach worth building, not a quick threshold tweak.")
