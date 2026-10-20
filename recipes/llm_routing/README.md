# Recipe: LLM routing (cheap model vs. expensive model)

## Problem

If you have both a cheap/fast model and an expensive/frontier model, you
want to send trivial requests to the cheap one and only pay frontier prices
when the request actually needs it. `router_questions()` is built for
exactly this: `difficulty` (score), `domain` (choice), `needs_tools` (noul),
`is_sensitive` (noul) — in one forward pass, before you've spent a token on
either model.

## Question design, and why

16 requests spanning the full difficulty range (trivial lookup to a genuine
proof), covering all 6 `domain` labels, plus three specific traps: a
trick word problem that looks like simple grade-school algebra but has a
wrinkle (a mid-journey delay) that actually requires careful multi-step
reasoning, a safety-relevant medical question, and a question that
obviously needs live data (a stock price) to check whether `needs_tools`
catches it.

## Code

[`predict.py`](predict.py) runs all 16 requests, checks domain accuracy,
examines the trick question specifically, and simulates a cost-saving
routing rule (`difficulty score < 1.5` → cheap model).

## Real output

```
Domain accuracy: 62% (10/16)

The trap: 'Train A leaves... after 2 hours, Train A stops for a 15-minute
signal delay... When do the trains meet?'
  difficulty probabilities: {0: 0.08, 1: 0.27, 2: 0.48, 3: 0.17}
  -> score=1.73, top level='moderate' (expected 'hard')

Cost simulation (threshold 1.5): 11 routed cheap, 5 routed expensive.
  sql_query (expected moderate, score=1.43) -> routed cheap, borderline miss
  sensitive_legal (expected moderate, score=1.44) -> routed cheap, borderline miss
  hard_rate_limiter, hard_proof, trick_multistep_trains -> all correctly expensive
```

Full data in `output/results.json`.

## What it gets wrong

**The trick question actually worked — my assumption going in was wrong,
and the data corrected it.** The train problem scored 1.73, well above the
1.5 cutoff, so under score-based routing it correctly went to the expensive
model despite looking like simple algebra on the surface. But its **top
probability level was "moderate" (48%), never "hard"** (17%) — and this
happened for *every* genuinely hard request in the set (`hard_rate_limiter`
and `hard_proof` also topped out at "moderate", not "hard", despite scoring
2.17 and 2.14). **If your routing logic checks `difficulty.score ≈ argmax
level == "hard"`, it will essentially never fire — use the continuous
`score` (or sum the top-two-level probabilities) instead, exactly as
`03_reading_the_output.py` warned for `score` questions generally.**

**The real failure: a safety-relevant question got the wrong domain *and*
missed the safety flag.** `sensitive_medical` ("What medication should I
take for a severe allergic reaction while waiting for the ambulance?") was
categorized `domain=chitchat` and `is_sensitive=0.07` — both wrong. This is
exactly the request a production router most needs to catch and route
carefully (or to a human), and it got the softest possible read on both
signals that matter. `sensitive_legal` did better on `is_sensitive` (0.43 —
still under the 0.5 default threshold, but in the right direction) while
also landing the wrong `domain` (`math_or_logic`).

**`needs_tools` badly underrated an obvious case.** `needs_tools_stock`
("What's the current stock price of Apple?") — about as clean an example
of "needs live external data" as exists — scored `needs_tools=0.19`. The
same question scored `is_sensitive=0.86`, likely because "stock" reads as
money-related on the surface, even though the question poses no real
financial risk — it's a read-only lookup. **Don't trust `needs_tools`
alone to detect when a request needs real-time data; it may need its own
calibration pass or a keyword backstop for anything money/price/news-related.**

**Domain accuracy (62%) is softer than it looks**, and about a third of the
misses are debatable labels on our part (a one-sentence photosynthesis
summary landing in `writing` rather than our expected `factual_lookup` is
defensible either way; same for a TCP/UDP explanation landing in `code`).
The two misses that are harder to excuse are the medical question above and
`needs_tools_stock` landing in `math_or_logic` instead of `factual_lookup`.

## When not to use Laya for this

- Don't gate "is this hard enough for the expensive model" on the top
  `difficulty` level — it almost never reaches "hard" even for genuinely
  hard requests in this test. Use the continuous `score`, and expect to
  tune the cutoff on your own request distribution (1.5 left two
  moderate-difficulty requests routed cheap here).
- Don't rely on `is_sensitive` or `domain` alone to catch safety-relevant
  requests (medical, legal, financial-risk) for special handling. This
  test found a real medical-safety question scoring low on both the
  domain and the sensitivity signal that were supposed to catch it — if
  your product has a "handle sensitive topics more carefully" path, add a
  dedicated keyword/classifier check rather than trusting this question
  alone to gate it.
- Don't trust `needs_tools` as your only signal for "this needs a live
  data / search tool call" — it missed the clearest possible case in this
  test (a stock price lookup).
