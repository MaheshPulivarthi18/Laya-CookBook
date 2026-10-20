# Recipe: conversation scoring

## Problem

Support conversations are multi-turn, and the signal you care about (a
churn threat, rising frustration, whether the issue actually got resolved)
can appear anywhere in the exchange, not just the latest message. Laya's
own docstring says `state` can be "a text string, JSON dict, or
**conversation turn list**" — but nothing in the package's own examples
demonstrates the list form, and nothing confirms whether it actually reads
early turns or just the most recent one.

## Question design, and why

Part 1 checks **state format sensitivity**: the exact same 5-turn
conversation represented three ways (a raw list of turn dicts, that list
wrapped in `{"conversation": [...]}`, and a manually flattened
`"role: text"` transcript string) — does the representation change the
answer?

Part 2 runs 4 conversations through 4 custom questions (`churn_risk`,
`escalate`, `resolved`, `frustration`) specifically designed to separate
"reads the whole conversation" from "only reads the last message": one
conversation buries an explicit churn threat in turn 3 of 5 (not the
last turn), and another puts strong frustration *only* in the final turn
with no earlier buildup, as a contrasting case.

## Code

[`predict.py`](predict.py) runs both parts and prints every answer.

## Real output

```
Part 1 (same conversation, 3 state formats):
  raw list          -> churn_risk=0.7612  tokens=173
  wrapped dict       -> churn_risk=0.7384  tokens=178
  flattened string   -> churn_risk=0.8545  tokens=131   <- fewer tokens, higher score

Part 2:
  resolved_calm          churn_risk=0.070  escalate=0.105  frustration=1.08  resolved=0.763
  threat_buried_midway   churn_risk=0.761  escalate=0.759  frustration=2.14  resolved=0.204
  simple_resolved        churn_risk=0.148  escalate=0.080  frustration=0.37  resolved=0.497
  frustration_at_end     churn_risk=0.425  escalate=0.155  frustration=2.69  resolved=0.257

churn_risk accuracy: 75%   escalate accuracy: 75%
```

Full numbers in `output/results.json`.

## What it gets wrong (and one thing it gets right that I expected to fail)

**It does read the whole conversation, not just the last message — this
is the main thing I was testing for, and it held up.**
`threat_buried_midway`'s explicit churn threat ("if this happens again I'm
switching to your competitor") sits in turn 3 of 5, two turns before the
end. It was caught correctly (`churn_risk=0.761`, `escalate=0.759`). If
Laya only attended to the final turn, this would have scored low, since
the last message in that conversation is just "please fix it" — mildly
annoyed, no threat. It didn't fail that way.

**State format measurably changes the answer, and the cheapest format won.**
The flattened plain-text transcript scored *both* the highest confidence
(0.8545 vs. 0.76/0.74) *and* used fewer tokens (131 vs. 173/178) than
passing structured turn objects. The JSON structure (role keys, braces,
quotes) adds token overhead without adding signal here. **Prefer a simple
`"role: text"` flattened transcript over passing raw turn dicts/lists**,
for both cost and — on this one example — accuracy.

**A real internal inconsistency, not just a wrong answer: the most
frustrated, least-resolved conversation got one of the LOWEST escalation
scores.** `frustration_at_end` scored `frustration=2.69` — the highest of
all 4 conversations, close to the "very angry" ceiling — and `resolved=0.257`,
essentially tied for the lowest. Logically, "most angry and least resolved"
should be a strong escalation candidate. Instead `escalate=0.155`, barely
above `resolved_calm`'s 0.105 (a conversation that ended with the customer
thanking the agent). **Two signals computed in the very same forward pass
disagree with each other in a way that doesn't hold up under its own
logic** — `frustration` and `resolved` both correctly read this
conversation as bad, but `escalate` didn't act on that read. Don't assume
a high `frustration` + low `resolved` combination will be reflected in
`escalate`; check it explicitly rather than trusting one "should imply"
the other.

**One honest note on my own test design:** I originally expected
`frustration_at_end`'s `churn_risk` to also score high (I labeled it
`expected_churn_risk: true`), and marked it a miss at 0.425. On reflection
that label may have been too strict on my part — the customer expresses
real anger but never actually threatens to leave or cancel, unlike
`threat_buried_midway`'s explicit "switching to your competitor." A
`churn_risk` of 0.425 (leaning "no," but not confidently) might be a
reasonable read of "angry but no stated intent to leave," not a clear
model failure. The `escalate` inconsistency above is the finding I'd
actually stand behind; the `churn_risk` "miss" is murkier and I'm flagging
that rather than quietly claiming a clean failure.

## When not to use Laya for this

- Don't pass conversation turns as raw structured objects (lists of dicts
  with role/text keys) by default — a flattened text transcript was
  cheaper and scored better on this test. Verify on your own data, but
  default to flattening.
- Don't assume `frustration` and `resolved` scoring badly will be reflected
  in `escalate` — this test found a case where it wasn't. If your escalation
  logic matters, check `escalate` directly, or combine signals yourself
  (e.g. `escalate_flag = escalate >= T or (frustration >= 2 and resolved < 0.3)`)
  rather than trusting `escalate` alone to synthesize the others.
- Label your own test sets carefully when "frustration" and "churn risk" are
  conflated — anger and an explicit threat to leave are different signals,
  and this recipe's own design blurred that distinction in one case.
