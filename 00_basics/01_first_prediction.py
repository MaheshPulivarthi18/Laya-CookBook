"""01 - Your first Laya prediction.

Laya answers fixed-choice questions about a piece of text. You are not
prompting a chatbot: you are asking one or more typed questions, and you get
back probabilities, not prose.

Run:
    python 00_basics/01_first_prediction.py
"""
import os

os.environ.setdefault("USE_TF", "0")  # avoids a known TensorFlow-loading hang on import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router

# 1. The state: whatever you want a decision about. Any dict works; the
#    field names are yours to choose, as long as `instructions` below refers
#    to them correctly.
state = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan.",
}

# 2. The question: one entry in a dict. `type` fixes the answer shape,
#    `instructions` tells the model what to look for, `criteria` gives the
#    fixed set of possible answers (here, 4 labeled options).
questions = {
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
}

# 3. Router picks the right checkpoint for the input's script/language and
#    runs it. `device="mps"` uses the Apple GPU; drop it to let torch pick.
router = Router(device="mps")
result = router.predict(state, questions)

# 4. The result. `answers[qid]` holds one entry per question you asked.
answer = result["answers"]["department"]
print("Chosen department :", answer["choice"])
print("Confidence         :", answer["confidence"])
print("Full distribution  :", answer["probabilities"])
print()
print("Which checkpoint answered  :", result["routing"]["model"])
print("Why                        :", result["routing"]["reason"])
print("Tokens the model actually saw:", result["usage"]["input_tokens"])
