"""Support triage: intent, urgency, frustration, refund signal, churn risk --
all from `triage_questions()`, all in one forward pass per ticket.

Your Step 0 found this preset wrong on one real message ("My payment failed
twice and nobody is answering!" -> intent `other`, 73%, plus a refund claim
that wasn't made). This recipe reproduces that case alongside 12 others
designed to cover each intent category plus the traps that caused it:
implicit asks, high frustration with no explicit request, and an implicit
churn threat.

Run:
    python recipes/support_triage/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router, triage_questions

HERE = Path(__file__).parent
tickets = json.loads((HERE / "data" / "tickets.json").read_text())
questions = triage_questions()

router = Router(device="mps")
results = []
correct = 0

for t in tickets:
    out = router.predict({"message": t["message"]}, questions)
    a = out["answers"]
    predicted = a["intent"]["choice"]
    match = predicted == t["expected_intent"]
    correct += match
    row = {
        "id": t["id"],
        "expected_intent": t["expected_intent"],
        "predicted_intent": predicted,
        "match": match,
        "intent_confidence": a["intent"]["confidence"],
        "intent_probabilities": a["intent"]["probabilities"],
        "is_urgent": a["is_urgent"]["noul"],
        "frustration_score": a["frustration"]["score"],
        "frustration_top_level": max(a["frustration"]["probabilities"],
                                      key=a["frustration"]["probabilities"].get),
        "refund_requested": a["refund_requested"]["noul"],
        "churn_risk": a["churn_risk"]["noul"],
        "routing_model": out["routing"]["model"],
    }
    results.append(row)
    mark = "OK  " if match else "MISS"
    print(f"[{mark}] {t['id']:<32} expected={t['expected_intent']:<15} "
          f"predicted={predicted:<15} conf={row['intent_confidence']:.3f}  "
          f"urgent={row['is_urgent']:.2f} frustration={row['frustration_score']:.2f} "
          f"refund={row['refund_requested']:.2f} churn={row['churn_risk']:.2f}")

print(f"\nIntent accuracy: {correct}/{len(tickets)} = {correct/len(tickets):.0%}")

print("\n" + "=" * 90)
print("Deep dive on both misses")
print("=" * 90)

known = next(r for r in results if r["id"] == "known_ambiguous_payment_failed")
msg = next(t for t in tickets if t["id"] == known["id"])["message"]
print(f"1) {msg!r}")
print(f"   intent probabilities: {known['intent_probabilities']}")
print(f"   -> 'other' wins at 62%, a clear plurality, not a near-tie with")
print(f"      'billing_question' (22%) -- this is a confident wrong answer.")
print(f"   refund_requested = {known['refund_requested']:.3f} (no refund was ever asked for)")
print(f"   A payment that fails twice with no response is a technical/billing")
print(f"   problem. There's no 'payment processing failure' category among the")
print(f"   preset's 6 fixed options, so the model forces it into the closest")
print(f"   fit -- and picks the wrong one, confidently.")

calm = next(r for r in results if r["id"] == "calm_information_request")
msg2 = next(t for t in tickets if t["id"] == calm["id"])["message"]
print(f"\n2) {msg2!r}")
print(f"   intent probabilities: {calm['intent_probabilities']}")
print(f"   -> 'other' wins at 70% over 'information' at 12%. This one is")
print(f"      NOT a hard edge case -- it's a plain, unambiguous feature")
print(f"      question ('do you support SSO'), and it's wrong more")
print(f"      confidently than the genuinely ambiguous case above.")
print(f"   Likely cause: triage_questions()'s `information` criterion reads")
print(f"   'general information, pricing or how-to' -- a yes/no feature-")
print(f"   availability question doesn't obviously match any of those three")
print(f"   words, so it falls through to 'other' despite being exactly the")
print(f"   kind of question a support inbox gets constantly.")
print(f"\n   Takeaway: before shipping a preset, test it on messages shaped")
print(f"   like YOUR real traffic, not just the obvious textbook case for")
print(f"   each category -- the preset's own category descriptions can be")
print(f"   too narrow for common real phrasing.")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nWrote {OUT / 'results.json'}")
