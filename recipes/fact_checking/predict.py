"""Fact checking / hallucination detection, using `LayaEvaluator` -- the one
class in the hidden API whose docstring literally says "hallucination
evaluation" and that we hadn't actually used until now.

`LayaEvaluator.evaluate_strings(prediction=..., input=...)` builds the state
as `{"input": input, "prediction": prediction}` for you (see
laya/integrations/langchain.py). So we write ONE noul question that reads
both fields: is `prediction` fully supported by `input`, with nothing added
or contradicted? 12 (context, claim) pairs: direct support, paraphrase,
negation, and five different flavors of hallucination (fabricated detail,
contradiction, wrong entity, true-but-not-in-context, invented causation,
wrong date).

Run:
    python recipes/fact_checking/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import LayaEvaluator

HERE = Path(__file__).parent
claims = json.loads((HERE / "data" / "claims.json").read_text())

hallucination_q = {
    "supported": {
        "type": "noul",
        "instructions": "Is every fact in `prediction` explicitly stated or directly implied "
                         "by `input`, with nothing added, changed, or contradicted?",
        "criteria": {
            "true": "prediction is fully supported by input -- a faithful restatement or "
                    "accurate summary",
            "false": "prediction adds a detail not in input, contradicts input, or states "
                     "something true in general but not actually present in input",
        },
    }
}
evaluator = LayaEvaluator(questions=hallucination_q)

results = []
for c in claims:
    answers = evaluator.evaluate_strings(prediction=c["prediction"], input=c["input"])
    supported_noul = answers["supported"]["noul"]
    predicted = "supported" if supported_noul >= 0.5 else "unsupported"
    match = predicted == c["label"]
    results.append({
        "id": c["id"], "label": c["label"], "predicted": predicted, "match": match,
        "supported_noul": supported_noul, "confidence": answers["supported"]["confidence"],
    })
    mark = "OK  " if match else "MISS"
    print(f"[{mark}] {c['id']:<36} label={c['label']:<11} predicted={predicted:<11} "
          f"supported_noul={supported_noul:.3f}")

acc = sum(r["match"] for r in results) / len(results)
print(f"\nAccuracy: {acc:.0%}")

tp = sum(1 for r in results if r["label"] == "supported" and r["predicted"] == "supported")
fn = sum(1 for r in results if r["label"] == "supported" and r["predicted"] == "unsupported")
tn = sum(1 for r in results if r["label"] == "unsupported" and r["predicted"] == "unsupported")
fp = sum(1 for r in results if r["label"] == "unsupported" and r["predicted"] == "supported")
print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
print("FP here means a HALLUCINATED claim was marked supported -- the dangerous direction")
print("for a fact-checking guard, since it would let a fabricated claim through unflagged.")

print()
print("=" * 90)
print("Where it fails")
print("=" * 90)
for r, c in zip(results, claims):
    if not r["match"]:
        print(f"{r['id']}:")
        print(f"  input:      {c['input']}")
        print(f"  prediction: {c['prediction']}")
        print(f"  supported_noul={r['supported_noul']:.3f} confidence={r['confidence']:.3f}")

print()
print("=" * 90)
print("Threshold sweep -- is 0.5 actually the right cutoff for THIS question?")
print("=" * 90)
for t in (0.5, 0.6, 0.7, 0.75, 0.8):
    acc_t = sum(
        1 for r in results
        if (r["supported_noul"] >= t) == (r["label"] == "supported")
    ) / len(results)
    print(f"  threshold={t:.2f} -> accuracy={acc_t:.0%}")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nWrote {OUT / 'results.json'}")
