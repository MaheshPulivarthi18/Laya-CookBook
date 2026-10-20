"""Email routing: classify inbound email into a department, flag spam/phishing,
score urgency, and detect whether a reply is expected -- all in one forward
pass per email, using the `email_questions()` preset.

Run:
    python recipes/email_routing/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router, email_questions

HERE = Path(__file__).parent
emails = json.loads((HERE / "data" / "emails.json").read_text())
questions = email_questions()

router = Router(device="mps")
results = []
correct = 0

for email in emails:
    state = {"subject": email["subject"], "body": email["body"]}
    out = router.predict(state, questions)
    answers = out["answers"]
    predicted = answers["category"]["choice"]
    is_match = predicted == email["expected_category"]
    correct += is_match

    row = {
        "id": email["id"],
        "expected": email["expected_category"],
        "predicted": predicted,
        "match": is_match,
        "category_confidence": answers["category"]["confidence"],
        "category_probabilities": answers["category"]["probabilities"],
        "is_spam": answers["is_spam"]["noul"],
        "is_phishing": answers["is_phishing"]["noul"],
        "urgency_score": answers["urgency"]["score"],
        "needs_reply": answers["needs_reply"]["noul"],
        "routing_model": out["routing"]["model"],
        "input_tokens": out["usage"]["input_tokens"],
    }
    results.append(row)

    mark = "OK  " if is_match else "MISS"
    print(f"[{mark}] {email['id']:<32} expected={email['expected_category']:<10} "
          f"predicted={predicted:<10} conf={row['category_confidence']:.3f}  "
          f"spam={row['is_spam']:.2f} phish={row['is_phishing']:.2f} "
          f"urgency={row['urgency_score']:.2f} model={row['routing_model']}")

print()
print(f"Category accuracy: {correct}/{len(emails)} = {correct / len(emails):.0%}")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Wrote {OUT / 'results.json'}")
