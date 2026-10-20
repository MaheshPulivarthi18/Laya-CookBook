# Recipe: fact checking / hallucination detection

## Problem

After an LLM answers using some retrieved source text, you want a cheap
check: does the answer's claim actually say what the source says, or did
the model add, change, or invent something? This is the specific use case
`LayaEvaluator`'s docstring names ("hallucination evaluation") but that
nothing in the package's examples actually demonstrates.

## Question design, and why

`LayaEvaluator.evaluate_strings(prediction=..., input=...)` builds the state
as `{"input": input, "prediction": prediction}` for you — see
`laya/integrations/langchain.py`. We write one `noul` question that reads
both fields: *"Is every fact in `prediction` explicitly stated or directly
implied by `input`, with nothing added, changed, or contradicted?"*

The 12-pair test set covers supported claims (direct restatement,
paraphrase, negation, summary, quote) and five distinct hallucination
shapes: a fabricated extra detail, a scale contradiction, an added
capability, a swapped name, a claim that's true in the real world but not
actually stated in the given source, an invented causal explanation, and a
swapped date.

## Code

[`predict.py`](predict.py) runs all 12 pairs through `LayaEvaluator`, scores
accuracy at the default 0.5 threshold, then sweeps the threshold to check
whether 0.5 is actually the right operating point for this question.

## Real output

```
Accuracy at threshold 0.5: 75% (9/12)
TP=5 FN=0 TN=4 FP=3   <- all 3 misses are hallucinations marked "supported"

Misses, all three:
  contradicted_scale:        45 min/EU-only  -> "several hours...all regions"  noul=0.638
  unsupported_wrong_entity:  "Alice approved" -> "Bob approved"                noul=0.651
  unsupported_wrong_date:    September 2026   -> "October 2026"               noul=0.640

Threshold sweep:
  0.50 -> 75%
  0.60 -> 75%
  0.70 -> 100%
  0.75 -> 100%
  0.80 -> 100%
```

Full numbers in `output/results.json`.

## What it gets wrong

**Every false positive is the same shape: a single swapped entity or value
inside an otherwise-identical sentence.** A name swap (Alice→Bob), a date
swap (September→October), a scale/duration swap (45 minutes, EU-only →
several hours, all regions) — all three scored right around 0.64–0.65,
just over the default 0.5 threshold. Meanwhile every claim that **added**
unsupported content (a fabricated 30%-enterprise-growth detail, an invented
causal explanation, an extra export format, a true-but-out-of-context
population fact) was caught cleanly, at noul ≤0.028. **The failure mode is
specific: the question recognizes topical/structural overlap with the
source well, but doesn't reliably verify that the specific entity values
inside a matching sentence are actually the same ones.** This is a known
weak spot for this style of entailment check generally, not unique to
Laya — but it's exactly the error pattern (one changed number, name, or
date) that matters most in a real fact-checking guard, since it's the
hardest kind of hallucination for a human skimmer to catch too.

**The fix that actually worked, verified, not assumed: raise the
threshold to ≥0.7.** None of the misses were confidently wrong — their
scores (0.638–0.651) sit much closer to 0.5 than to 1.0, while every
genuinely supported claim scored ≥0.899. A threshold sweep (printed by the
script) confirms accuracy jumps from 75% to 100% at 0.70 and holds through
0.80. **This is a single 12-example test set, not a calibration study** —
don't copy "0.7" into production without recalibrating on your own
hallucination examples (see the planned `production/calibration.py`) — but
it demonstrates the principle directly: for support-checking specifically,
treat anything below ~0.7, not just below 0.5, as "needs review," because
the entity-swap failure mode clusters right above the default cutoff.

## When not to use Laya for this

- Don't use the default 0.5 threshold for a support/entailment-style noul
  question without checking where your own false positives cluster first —
  here they sat at 0.64, comfortably "yes" under a naive threshold, and
  would have silently let three fabricated claims through.
- Don't trust this check alone to catch a wrong number, name, or date
  swapped into an otherwise-faithful sentence, even after raising the
  threshold on this test set — verify the specific numbers/names/dates in a
  high-stakes claim with a separate, exact string/value comparison against
  the source rather than relying on the model's semantic judgment alone.
- If your source documents are long (near the 512/1024 token limit), check
  `usage.input_tokens` — this recipe's examples are short by design,
  real sources often are not, and silent truncation would mean the model is
  fact-checking against a source it never fully saw.
