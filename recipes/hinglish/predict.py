"""Hinglish: code-mixed Hindi-English, the case 00_basics/04 found the
router's script/stopword heuristic handles inconsistently -- sometimes
landing on `english`, sometimes `multilingual`, sometimes falling back to
the default depending on exact wording.

This recipe doesn't just look at routing -- it tests ACCURACY under three
conditions for 6 triage messages and 2 guard messages, all genuinely
code-mixed (not pure Hindi, not pure English): auto-routed (whatever the
heuristic picks), forced onto `english`, and forced onto `multilingual`.
That answers the practical question: if you have real Hinglish traffic,
which checkpoint should you actually pin it to?

Run:
    python recipes/hinglish/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router, guard_questions, triage_questions

HERE = Path(__file__).parent
data = json.loads((HERE / "data" / "messages.json").read_text())
router = Router(device="mps")
MODELS = [None, "english", "multilingual"]  # None = auto


def label(m):
    return "auto" if m is None else m


print("=" * 90)
print("Triage messages (intent accuracy under each forced model)")
print("=" * 90)
triage_q = triage_questions()
triage_rows = []
for msg in data["triage"]:
    row = {"id": msg["id"], "expected": msg["expected_intent"], "results": {}}
    for model in MODELS:
        out = router.predict({"message": msg["message"]}, triage_q, model=model)
        predicted = out["answers"]["intent"]["choice"]
        row["results"][label(model)] = {
            "predicted": predicted,
            "match": predicted == msg["expected_intent"],
            "confidence": out["answers"]["intent"]["confidence"],
            "actual_model_used": out["routing"]["model"],
        }
    triage_rows.append(row)
    print(f"\n{msg['id']} (expected: {msg['expected_intent']})")
    for model in MODELS:
        r = row["results"][label(model)]
        mark = "OK  " if r["match"] else "MISS"
        print(f"  [{mark}] {label(model):<10} -> {r['predicted']:<15} "
              f"conf={r['confidence']:.3f}  (actually ran on: {r['actual_model_used']})")

print("\n" + "=" * 90)
print("Guard messages (jailbreak/phishing recall under each forced model)")
print("=" * 90)
guard_q = guard_questions()
guard_rows = []
for msg in data["guard"]:
    row = {"id": msg["id"], "label": msg["label"], "results": {}}
    for model in MODELS:
        out = router.predict({"prompt": msg["prompt"]}, guard_q, model=model)
        jb = out["answers"]["jailbreak"]["noul"]
        inj = out["answers"]["prompt_injection"]["noul"]
        row["results"][label(model)] = {
            "jailbreak": jb, "prompt_injection": inj,
            "caught": jb >= 0.5 or inj >= 0.5,
            "actual_model_used": out["routing"]["model"],
        }
    guard_rows.append(row)
    print(f"\n{msg['id']} ({msg['label']})")
    for model in MODELS:
        r = row["results"][label(model)]
        mark = "OK  " if r["caught"] else "MISS"
        print(f"  [{mark}] {label(model):<10} -> jailbreak={r['jailbreak']:.3f} "
              f"injection={r['prompt_injection']:.3f}  (actually ran on: {r['actual_model_used']})")

print("\n" + "=" * 90)
print("Summary: accuracy by forced model")
print("=" * 90)
for model in MODELS:
    triage_acc = sum(r["results"][label(model)]["match"] for r in triage_rows) / len(triage_rows)
    guard_acc = sum(r["results"][label(model)]["caught"] for r in guard_rows) / len(guard_rows)
    print(f"  {label(model):<10} triage_intent_accuracy={triage_acc:.0%}  guard_recall={guard_acc:.0%}")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(
    json.dumps({"triage": triage_rows, "guard": guard_rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"\nWrote {OUT / 'results.json'}")
