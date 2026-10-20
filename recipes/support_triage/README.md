# Recipe: support triage

## Problem

Every inbound support message needs the same quick decisions before a human
even reads it: what does the customer want (`intent`), is it urgent, how
frustrated are they, are they asking for money back, and are they at risk
of leaving. `triage_questions()` answers all five in one forward pass.

## Question design, and why

We reuse the preset as-is (expects state key `message`), and build a 13-ticket
test set that covers each of the 6 intent categories cleanly, plus the traps
that caused your Step 0 finding: an implicit refund ask, high frustration
with no explicit request, an implicit churn threat phrased as a deadline,
and the exact ambiguous message from Step 0 ("My payment failed twice and
nobody is answering!") so we can go deeper than one example.

## Code

[`predict.py`](predict.py) runs all 13 tickets, scores intent accuracy, and
prints a deep dive on both misses with full probability breakdowns.

## Real output

```
[OK  ] refund_clear                     expected=refund           predicted=refund           conf=0.995
[OK  ] technical_help_clear             expected=technical_help   predicted=technical_help   conf=0.983
[OK  ] billing_question_clear           expected=billing_question predicted=billing_question conf=0.989
[OK  ] information_clear                expected=information      predicted=information      conf=0.639
[OK  ] cancellation_clear               expected=cancellation     predicted=cancellation     conf=0.989
[OK  ] other_clear                      expected=other            predicted=other            conf=0.221
[MISS] known_ambiguous_payment_failed   expected=technical_help   predicted=other            conf=0.419
[OK  ] implicit_refund_angry            expected=refund           predicted=refund           conf=0.994
[OK  ] high_frustration_no_clear_ask    expected=other            predicted=other            conf=0.348
[OK  ] implicit_churn_threat            expected=technical_help   predicted=technical_help   conf=0.703
[OK  ] urgent_deadline                  expected=technical_help   predicted=technical_help   conf=0.839
[MISS] calm_information_request        expected=information      predicted=other            conf=0.414
[OK  ] hindi_cancellation_request       expected=cancellation     predicted=cancellation     conf=0.999

Intent accuracy: 11/13 = 85%
```

Full probabilities for every ticket are in `output/results.json`.

## What it gets wrong

**Miss 1 — "My payment failed twice and nobody is answering!" → `other`
(62%), not `billing_question` (22%).** This is a confident wrong answer, not
a near-tie — and `refund_requested` fires at 0.73 even though no refund was
asked for. The preset's 6 categories have no "payment processing failure"
bucket, so a real, common support scenario gets forced into the worst-fit
option.

**Miss 2 — a plain SSO availability question → `other` (70%), not
`information` (12%).** This is the more important miss: it's not an edge
case. "Do you support SSO login" is an ordinary, unambiguous information
request, and it loses to `other` by a wider margin than the genuinely
ambiguous payment case above. The likely cause: `triage_questions()`
describes `information` as "general information, pricing or how-to" — a
yes/no feature-availability question doesn't read as any of those three
words, so it falls through. **The lesson isn't "this preset struggles with
hard cases" — it's "test the preset's category descriptions against your
actual common traffic, because an everyday question can miss a category
whose description just doesn't happen to cover it."**

**What worked better than expected:** `implicit_churn_threat` ("if this
isn't fixed by Friday, we're moving to a competitor") correctly triggered
`churn_risk=0.83` and `is_urgent=0.86` and classified as `technical_help` —
the model read the conditional threat correctly without an explicit "I want
to cancel." Same for `implicit_refund_angry`, which correctly separated
`refund_requested=0.97` from `churn_risk=0.66` even though both signals are
present in one angry message.

## When not to use Laya for this

- If your support categories don't map cleanly onto the preset's 6 labels
  (refund / technical_help / billing_question / information / cancellation /
  other) — and "payment processing failure" above shows that's easy to miss
  — add your own `criteria` rather than trusting the preset's defaults.
  Validate every category description against examples of what your real
  customers actually write, not just a textbook phrasing of each intent.
- Don't rely on `intent` confidence alone to flag "needs human review" —
  Miss 2 scored confidence 0.414, in the same range as several *correct*
  answers (`other_clear` at 0.221, `high_frustration_no_clear_ask` at
  0.348). Confidence shrinks with the option count (`03_reading_the_output.py`)
  and doesn't reliably separate "ambiguous and right" from "wrong."
