"""Conversation scoring: Laya's own docstrings say `state` can be a string,
a dict, OR a conversation turn list (`Agent.predict`: "state: Text string,
JSON dict, or conversation turn list") -- but nothing in the package's
examples actually uses the list form. This recipe does, and checks the
thing that actually matters: does it read the WHOLE conversation, or just
the most recent message?

Part 1 checks three ways of representing the same conversation (raw list,
wrapped dict, manually flattened string) for consistency.
Part 2 runs 4 conversations through custom questions, including one where
a churn threat is buried in the MIDDLE of the conversation (not the last
turn) specifically to test whether early turns are actually read.

Run:
    python recipes/conversation_scoring/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router

HERE = Path(__file__).parent
conversations = json.loads((HERE / "data" / "conversations.json").read_text())
router = Router(device="mps")

questions = {
    "churn_risk": {"type": "noul",
        "instructions": "At any point in this conversation, does the customer signal they "
                         "might cancel or switch to a competitor?"},
    "escalate": {"type": "noul",
        "instructions": "Based on the full conversation, should this be escalated to a "
                         "human supervisor?"},
    "resolved": {"type": "noul",
        "instructions": "Was the customer's issue resolved by the end of the conversation?"},
    "frustration": {"type": "score",
        "instructions": "How frustrated is the customer by the end of the conversation?",
        "criteria": ["calm", "concerned", "annoyed", "very angry"]},
}

print("=" * 90)
print("Part 1: does the STATE FORMAT matter for the same conversation?")
print("=" * 90)
sample = conversations[1]  # threat_buried_midway
raw_list = sample["turns"]
wrapped_dict = {"conversation": sample["turns"]}
flattened_string = "\n".join(f"{t['role']}: {t['text']}" for t in sample["turns"])

for label, state in [("raw list", raw_list), ("wrapped dict", wrapped_dict),
                      ("flattened string", flattened_string)]:
    out = router.predict(state, {"churn_risk": questions["churn_risk"]})
    a = out["answers"]["churn_risk"]
    print(f"  {label:<20} -> churn_risk={a['noul']:.4f}  tokens={out['usage']['input_tokens']}")

print("\n" + "=" * 90)
print("Part 2: 4 conversations, full question set (using the raw list form)")
print("=" * 90)
results = []
for conv in conversations:
    out = router.predict(conv["turns"], questions)
    a = out["answers"]
    row = {
        "id": conv["id"],
        "churn_risk": a["churn_risk"]["noul"],
        "escalate": a["escalate"]["noul"],
        "resolved": a["resolved"]["noul"],
        "frustration_score": a["frustration"]["score"],
        "expected_churn_risk": conv["expected_churn_risk"],
        "expected_escalate": conv["expected_escalate"],
        "churn_match": (a["churn_risk"]["noul"] >= 0.5) == conv["expected_churn_risk"],
        "escalate_match": (a["escalate"]["noul"] >= 0.5) == conv["expected_escalate"],
        "input_tokens": out["usage"]["input_tokens"],
    }
    results.append(row)
    cm = "OK  " if row["churn_match"] else "MISS"
    em = "OK  " if row["escalate_match"] else "MISS"
    print(f"\n{conv['id']} ({len(conv['turns'])} turns)")
    print(f"  [{cm}] churn_risk={row['churn_risk']:.3f} (expected {conv['expected_churn_risk']})")
    print(f"  [{em}] escalate={row['escalate']:.3f} (expected {conv['expected_escalate']})")
    print(f"       resolved={row['resolved']:.3f}  frustration={row['frustration_score']:.2f}")

acc_churn = sum(r["churn_match"] for r in results) / len(results)
acc_escalate = sum(r["escalate_match"] for r in results) / len(results)
print(f"\nchurn_risk accuracy: {acc_churn:.0%}   escalate accuracy: {acc_escalate:.0%}")

print("\n" + "=" * 90)
print("The key comparison: threat buried mid-conversation vs. frustration at the end")
print("=" * 90)
mid = next(r for r in results if r["id"] == "threat_buried_midway")
end = next(r for r in results if r["id"] == "frustration_at_end")
print(f"threat_buried_midway (threat in turn 3 of 5, NOT the last turn): "
      f"churn_risk={mid['churn_risk']:.3f}")
print(f"frustration_at_end   (frustration IS the last turn):              "
      f"churn_risk={end['churn_risk']:.3f}")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nWrote {OUT / 'results.json'}")
