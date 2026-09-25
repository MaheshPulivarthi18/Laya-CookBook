"""03 - Reading the output: every field explained, confidence formulas proven.

This recipe doesn't just describe what `confidence` means for each question
type -- it recomputes it from the raw probabilities and checks the result
against what Laya actually returned, so you can see the formula is real.

Run:
    python 00_basics/03_reading_the_output.py
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math

from laya import Router


def entropy_confidence(probs):
    """What Laya uses for `choice` and `score`: 1 - H(p)/log(K)."""
    k = len(probs)
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return 1 - h / math.log(k)


def noul_confidence(p_true):
    """What Laya uses for `noul`: NOT the entropy formula above.
    Just how far the winning side is from a coin flip."""
    return max(p_true, 1 - p_true)


state = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan.",
}
questions = {
    "department": {"type": "choice",
        "instructions": "Which department should handle the request in `body`?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score",
        "instructions": "How urgent is the request in `body`?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul",
        "instructions": "Does the user in `body` threaten to cancel or leave?"},
}

result = Router(device="mps").predict(state, questions)

print("=" * 60)
print("choice / score -> confidence = 1 - entropy(p) / log(num_options)")
print("=" * 60)
for qid in ("department", "urgency"):
    a = result["answers"][qid]
    probs = list(a["probabilities"].values())
    computed = entropy_confidence(probs)
    print(f"{qid}: reported={a['confidence']:.4f}  recomputed={computed:.4f}")

print()
print("=" * 60)
print("noul -> confidence = max(p, 1 - p)   (NOT the entropy formula)")
print("=" * 60)
a = result["answers"]["churn_risk"]
computed = noul_confidence(a["noul"])
print(f"churn_risk: reported={a['confidence']:.4f}  recomputed={computed:.4f}")
would_be_entropy = entropy_confidence([a["noul"], 1 - a["noul"]])
print(f"  (if it used the entropy formula instead, confidence would be {would_be_entropy:.4f} -- different!)")

print()
print("=" * 60)
print("Every other field in a single answer")
print("=" * 60)
print("""
answer['type']          -- "choice" | "score" | "noul"
answer['choice'|'score'|'noul']
                         -- choice: argmax label.
                         -- score: probability-WEIGHTED MEAN of level indices,
                            not the most likely level -- read `probabilities`
                            for that.
                         -- noul: P(statement is true), a single float.
answer['probabilities'] -- full distribution (choice, score). Always sums to 1.
answer['legend']        -- score only: {index: level description}.
answer['confidence']    -- see the two formulas above. NOT comparable across
                            question types, and for `choice`/`score` it also
                            shrinks as the option/level count grows -- it is
                            not comparable across questions with different K
                            either.
answer['action']['act_probability']
                         -- always 1.0 in every test run against Laya 0.3.11.
                            Carries no information in this version -- do not
                            build decisions on it.
result['usage']['input_tokens']
                         -- if this equals the checkpoint's context limit
                            (512 english / 1024 multilingual & typed-decisions),
                            your text was SILENTLY TRUNCATED. Laya gives no
                            warning. A `noul` answer can be confidently wrong
                            (confidence 1.0) simply because the model never
                            saw the relevant sentence. Always check this
                            before trusting a long-text answer.
result['routing']       -- which checkpoint ran and why (see recipe 04).
""")
