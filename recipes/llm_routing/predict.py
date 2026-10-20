"""LLM routing: can `router_questions()` cheaply decide which requests need
an expensive model, before spending a token on one?

16 requests spanning trivial to hard, across all 6 domains, plus a
deliberate trap: a word problem that LOOKS like simple grade-school algebra
but has a wrinkle (a mid-journey delay) that actually requires careful
multi-step reasoning -- does `difficulty` catch that it's hard, or does it
score it as easy because it superficially resembles a simple problem?

Run:
    python recipes/llm_routing/predict.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laya import Router, router_questions

HERE = Path(__file__).parent
requests = json.loads((HERE / "data" / "requests.json").read_text())
questions = router_questions()
LEVEL_NAMES = ["trivial", "easy", "moderate", "hard"]

router = Router(device="mps")
results = []

for r in requests:
    out = router.predict({"request": r["request"]}, questions)
    a = out["answers"]
    row = {
        "id": r["id"],
        "expected_difficulty": r["expected_difficulty"],
        "difficulty_score": a["difficulty"]["score"],
        "difficulty_top_level": max(a["difficulty"]["probabilities"],
                                     key=a["difficulty"]["probabilities"].get),
        "difficulty_probabilities": a["difficulty"]["probabilities"],
        "expected_domain": r["expected_domain"],
        "domain": a["domain"]["choice"],
        "domain_match": a["domain"]["choice"] == r["expected_domain"],
        "needs_tools": a["needs_tools"]["noul"],
        "expected_needs_tools": r["expected_needs_tools"],
        "is_sensitive": a["is_sensitive"]["noul"],
        "expected_sensitive": r["expected_sensitive"],
    }
    results.append(row)
    dmatch = "OK  " if row["domain_match"] else "MISS"
    print(f"[{dmatch}] {r['id']:<26} difficulty={row['difficulty_score']:.2f} "
          f"(expected level {r['expected_difficulty']}={LEVEL_NAMES[r['expected_difficulty']]}, "
          f"top={row['difficulty_top_level']})  domain={row['domain']:<14} "
          f"tools={row['needs_tools']:.2f} sensitive={row['is_sensitive']:.2f}")

domain_acc = sum(r["domain_match"] for r in results) / len(results)
print(f"\nDomain accuracy: {domain_acc:.0%}")

print("\n" + "=" * 90)
print("The trap: does a deceptive-looking word problem get rated correctly hard?")
print("=" * 90)
trap = next(r for r in results if r["id"] == "trick_multistep_trains")
print(f"request: {next(x for x in requests if x['id'] == trap['id'])['request']!r}")
print(f"difficulty probabilities: {trap['difficulty_probabilities']}")
print(f"-> score={trap['difficulty_score']:.2f}, top level='{trap['difficulty_top_level']}' "
      f"(expected 'hard')")

print("\n" + "=" * 90)
print("Cost-saving simulation: route difficulty score < 1.5 to a cheap model")
print("=" * 90)
cheap, expensive = [], []
for row, req in zip(results, requests):
    bucket = cheap if row["difficulty_score"] < 1.5 else expensive
    bucket.append((row["id"], req["expected_difficulty"]))
print(f"Routed to CHEAP model ({len(cheap)}):")
for rid, exp in cheap:
    flag = " <- actually expected HARD/MODERATE, routed cheap anyway" if exp >= 2 else ""
    print(f"  {rid} (expected level {exp}){flag}")
print(f"\nRouted to EXPENSIVE model ({len(expensive)}):")
for rid, exp in expensive:
    flag = " <- actually expected TRIVIAL/EASY, routed expensive anyway" if exp <= 1 else ""
    print(f"  {rid} (expected level {exp}){flag}")

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nWrote {OUT / 'results.json'}")
