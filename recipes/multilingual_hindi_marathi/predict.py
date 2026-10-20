"""Multilingual support triage: does the SAME message, translated into
English, Hindi, and Marathi, get the SAME triage answers?

recipes/phishing_detection already found a language-driven bias on safety
(noul) questions. This recipe checks ordinary (non-adversarial)
classification instead -- intent, urgency, frustration -- using
triage_questions() on 5 real-feeling support messages, each held constant
in content and translated into 3 languages. 15 calls total, measuring both
per-language accuracy and cross-language CONSISTENCY on identical content.

Run:
    python recipes/multilingual_hindi_marathi/predict.py
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

LANGS = ["english", "hindi", "marathi"]
results = []

for t in tickets:
    row = {"content_id": t["content_id"], "expected_intent": t["expected_intent"], "by_lang": {}}
    for lang in LANGS:
        out = router.predict({"message": t[lang]}, questions)
        a = out["answers"]
        row["by_lang"][lang] = {
            "predicted_intent": a["intent"]["choice"],
            "intent_confidence": a["intent"]["confidence"],
            "match": a["intent"]["choice"] == t["expected_intent"],
            "is_urgent": a["is_urgent"]["noul"],
            "frustration_score": a["frustration"]["score"],
            "routing_model": out["routing"]["model"],
            "input_tokens": out["usage"]["input_tokens"],
        }
    results.append(row)

    print(f"\n--- {t['content_id']} (expected intent: {t['expected_intent']}) ---")
    predicted_set = set()
    for lang in LANGS:
        d = row["by_lang"][lang]
        predicted_set.add(d["predicted_intent"])
        mark = "OK  " if d["match"] else "MISS"
        print(f"  [{mark}] {lang:<10} -> intent={d['predicted_intent']:<15} "
              f"conf={d['intent_confidence']:.3f}  urgent={d['is_urgent']:.2f} "
              f"frustration={d['frustration_score']:.2f}  model={d['routing_model']} "
              f"tokens={d['input_tokens']}")
    row["cross_lingual_consistent"] = len(predicted_set) == 1
    print(f"  cross-lingual consistent: {row['cross_lingual_consistent']} (predicted: {predicted_set})")

per_lang_acc = {
    lang: sum(r["by_lang"][lang]["match"] for r in results) / len(results)
    for lang in LANGS
}
consistent_count = sum(r["cross_lingual_consistent"] for r in results)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
for lang, acc in per_lang_acc.items():
    print(f"  {lang:<10} intent accuracy: {acc:.0%}")
print(f"  Cross-lingual consistency (same predicted intent in all 3 langs): "
      f"{consistent_count}/{len(results)}")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote {OUT / 'results.json'}")
