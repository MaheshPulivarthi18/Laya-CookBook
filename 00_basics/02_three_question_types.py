"""02 - The three question types: choice, score, noul.

Every Laya question is one of these three. This recipe asks all three about
the same email in a single forward pass, so you can compare their answer
shapes side by side.

Run:
    python 00_basics/02_three_question_types.py
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json

from laya import Router

state = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan.",
}

questions = {
    # choice: pick exactly one of N labeled, mutually exclusive options.
    # `criteria` is a dict {label: description}. Answer field is `choice`,
    # plus a full `probabilities` dict over every label.
    "department": {
        "type": "choice",
        "instructions": "Which department should handle the request in `body`?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    # score: a position on an ORDERED scale. `criteria` is a list, not a
    # dict -- order matters, index 0..N-1. Answer field is `score`, the
    # probability-WEIGHTED MEAN of the level indices (not the most likely
    # level -- read `probabilities` for that).
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the request in `body`?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    # noul: the probability that a yes/no statement is TRUE. No criteria
    # needed. Answer field is `noul`, a single number in [0, 1].
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user in `body` threaten to cancel or leave?",
    },
}

router = Router(device="mps")
result = router.predict(state, questions)

for qid, answer in result["answers"].items():
    print(f"--- {qid} ({answer['type']}) ---")
    print(json.dumps(answer, indent=2))
    print()

print("department: `choice` is just argmax(probabilities) -- the single most likely label.")
print("urgency   : `score` is a weighted mean across levels, not the most likely level.")
print("            Check `probabilities` if you need a discrete level.")
print("churn_risk: `noul` is one number -- P(the statement is true).")
