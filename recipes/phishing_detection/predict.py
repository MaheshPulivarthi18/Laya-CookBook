"""Phishing detection: does `is_phishing` hold up across languages?

recipes/email_routing found one alarming example: a legitimate Hindi
complaint flagged as 97% phishing. This recipe tests that properly instead
of trusting one example -- 16 messages, labeled phishing/legitimate,
including English, Hindi and Marathi, both directions (legit and phishing
in every language we test). It also compares the default `email_questions()`
criteria against a more detailed custom criteria, to check whether wording
(not just language) explains the gap.

Run:
    python recipes/phishing_detection/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router

HERE = Path(__file__).parent
messages = json.loads((HERE / "data" / "messages.json").read_text())

# The default, from email_questions() (laya/presets.py).
default_q = {
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
        "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"},
    }
}

# A more detailed version, testing whether richer criteria change the answer
# (your Step 0 found custom criteria shifted a legitimate email's phishing
# probability from 0.19 to 0.38 -- wording matters, so we check it here too).
detailed_q = {
    "is_phishing": {
        "type": "noul",
        "instructions": "Is `body` a phishing or scam message that tries to steal money, "
                         "passwords, card numbers or personal data -- as opposed to a normal "
                         "transactional notice, internal request, or genuine complaint?",
        "criteria": {
            "true": "asks the reader to click a link, call a number, or enter credentials/card "
                    "details under urgency or threat of suspension/loss",
            "false": "a routine notice, an internal work request, or a genuine customer complaint "
                     "that does not ask for credentials or payment details",
        },
    }
}


def script_of(text):
    # crude: any Devanagari codepoint present -> treat as Devanagari-script
    return "devanagari" if any("ऀ" <= ch <= "ॿ" for ch in text) else "latin"


def run(questions, label):
    router = Router(device="mps")
    rows = []
    for m in messages:
        out = router.predict({"body": m["text"]}, questions)
        ans = out["answers"]["is_phishing"]
        predicted = "phishing" if ans["noul"] >= 0.5 else "legitimate"
        rows.append({
            "id": m["id"],
            "true_label": m["label"],
            "script": script_of(m["text"]),
            "noul": ans["noul"],
            "confidence": ans["confidence"],
            "predicted": predicted,
            "correct": predicted == m["label"],
            "routing_model": out["routing"]["model"],
        })
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    for r in rows:
        mark = "OK  " if r["correct"] else "MISS"
        print(f"[{mark}] {r['id']:<32} true={r['true_label']:<11} script={r['script']:<10} "
              f"noul={r['noul']:.4f}  model={r['routing_model']}")
    by_script = {}
    for r in rows:
        by_script.setdefault(r["script"], []).append(r["correct"])
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nOverall accuracy: {acc:.0%}")
    for script, corrects in by_script.items():
        print(f"  {script:<10}: {sum(corrects)}/{len(corrects)} = {sum(corrects)/len(corrects):.0%}")
    false_positives = [r for r in rows if r["true_label"] == "legitimate" and r["predicted"] == "phishing"]
    print(f"False positives (legitimate flagged as phishing): {len(false_positives)}")
    for fp in false_positives:
        print(f"  - {fp['id']} (script={fp['script']}, noul={fp['noul']:.4f})")
    return rows


default_rows = run(default_q, "DEFAULT criteria (from email_questions())")
detailed_rows = run(detailed_q, "DETAILED criteria (custom wording)")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(
    json.dumps({"default": default_rows, "detailed": detailed_rows}, indent=2),
    encoding="utf-8",
)
print(f"\nWrote {OUT / 'results.json'}")
