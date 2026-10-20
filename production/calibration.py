"""Calibration: is a 0.9-confidence answer actually right 90% of the time?

`laya.ece_score` (Expected Calibration Error) is in the public API tour
(00_basics/06 lists it among the hidden features) but nothing in the
package's examples actually calls it. This module does: it reuses the
already-labeled, already-run results from recipes/phishing_detection/
(16 messages, ground truth + `confidence` saved to output/results.json)
to measure real ECE, split by script -- putting a number on the Latin vs.
Devanagari gap that recipe only described qualitatively.

ECE close to 0 means confidence is trustworthy (a 0.9-confidence batch of
answers is right about 90% of the time). Higher ECE means confidence is
miscalibrated -- which direction (over- or under-confident) isn't shown by
ECE alone, so this script also prints raw accuracy per confidence bucket.

Run (after recipes/phishing_detection/predict.py has produced its
output/results.json -- already committed in this repo):
    python production/calibration.py
"""
import json
from pathlib import Path

import numpy as np

from laya import ece_score

ROOT = Path(__file__).parent.parent
results_path = ROOT / "recipes" / "phishing_detection" / "output" / "results.json"
data = json.loads(results_path.read_text())["default"]  # the default-criteria run

print("=" * 90)
print("ECE on recipes/phishing_detection/'s real labeled results (16 messages)")
print("=" * 90)


def report(rows, label):
    conf = np.array([r["confidence"] for r in rows])
    correct = np.array([1 if r["correct"] else 0 for r in rows])
    if len(rows) < 3:
        print(f"  {label}: n={len(rows)} (too few examples for a stable ECE)")
        return
    ece = ece_score(conf, correct, bins=5)
    acc = correct.mean()
    print(f"  {label}: n={len(rows)}  accuracy={acc:.0%}  mean_confidence={conf.mean():.3f}  "
          f"ECE={ece:.3f}")


report(data, "all 16")
report([r for r in data if r["script"] == "latin"], "latin script (English)")
report([r for r in data if r["script"] == "devanagari"], "devanagari script (Hindi/Marathi)")

print()
print("=" * 90)
print("Accuracy by confidence bucket -- which direction is the miscalibration?")
print("=" * 90)
buckets = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
for lo, hi in buckets:
    bucket = [r for r in data if lo <= r["confidence"] < hi]
    if not bucket:
        continue
    acc = sum(r["correct"] for r in bucket) / len(bucket)
    print(f"  confidence in [{lo:.2f}, {hi:.2f}): n={len(bucket)}  "
          f"mean_confidence={sum(r['confidence'] for r in bucket)/len(bucket):.3f}  "
          f"actual_accuracy={acc:.0%}")
    if acc < lo:
        print(f"    -> OVERCONFIDENT: claims ~{lo:.0%}+ but only {acc:.0%} right")

print()
print("Takeaway: don't trust a single global confidence number as 'this is")
print("calibrated.' Compute ECE (or at minimum a confidence-vs-accuracy table")
print("like the one above) ON YOUR OWN labeled data, broken out by whatever")
print("dimension you suspect matters (language, script, question type) --")
print("recipes/phishing_detection/'s qualitative finding (Hindi worse than")
print("English) becomes a concrete, comparable number here.")
