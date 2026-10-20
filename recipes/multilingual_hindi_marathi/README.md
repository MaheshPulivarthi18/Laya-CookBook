# Recipe: multilingual support triage (Hindi, Marathi)

## Problem

`recipes/phishing_detection/` found a real language-driven bias, but only
on safety (`noul`) questions. Does the same gap show up on ordinary,
non-adversarial classification — the everyday job of routing a support
ticket? This recipe holds **content constant** and only changes the
language: 5 support messages, each translated into English, Hindi, and
Marathi, run through the same `triage_questions()` preset. 15 calls,
comparing both per-language accuracy and cross-lingual *consistency* on
identical content.

## Question design, and why

We reuse `triage_questions()` exactly as in `recipes/support_triage/`, so
results are directly comparable to that recipe's English-only numbers. The
5 messages cover one example of each non-"other" intent plus a pure
compliment (expected `other`) — chosen specifically because it contains
zero billing/refund/complaint vocabulary in any of the three languages, as
a clean check for "does positive sentiment get misread as a request."

## Code

[`predict.py`](predict.py) runs all 15 (content × language) combinations,
reports per-language intent accuracy, and flags whether all three languages
agreed on the same predicted intent for identical content.

## Real output

```
english    intent accuracy: 80%
hindi      intent accuracy: 60%
marathi    intent accuracy: 80%
Cross-lingual consistency (same intent in all 3 languages): 3/5

billing_refund, technical_crash, cancellation -> consistent across all 3 languages
information_request -> 3-WAY DISAGREEMENT: other (en) / billing_question (hi) / information (mr)
compliment -> "Thank you so much, your support team was amazing!"
  english -> other        (correct)
  hindi   -> refund, confidence 1.000   <- WRONG, and total certainty
  marathi -> refund, confidence 0.999   <- WRONG, and total certainty
```

Full probabilities in `output/results.json`.

## What it gets wrong

**The headline finding: a pure compliment was read as a refund request, at
total confidence, in both non-English languages.** Checked the raw
probabilities directly (not just the top choice): Hindi scored
`refund=1.0000` and every other option `0.0000`. Marathi scored
`refund=0.9999`. Neither translation contains any billing, money, or
complaint vocabulary — "Thank you so much, your support team was amazing!"
translates to an unambiguous, content-free compliment in both languages.
English correctly read it as `other` (though at low confidence, 0.247,
appropriately reflecting that "other" is a weak-fit catch-all even when
right). **This isn't a borderline miscalibration — it's complete, maximum-
confidence certainty assigned to the opposite of the correct answer**, and
it happened in both non-English languages tested, not just one.

**The second miss is a genuine 3-way disagreement, not just "non-English is
worse."** The information-request message ("Does this plan include
priority support?") got three *different* wrong-or-right answers across
three languages: English said `other`, Hindi said `billing_question`,
Marathi — the only one that got it right — said `information` but at low
confidence (0.343). English being wrong here matters: this rules out "it's
just a non-English problem," since the baseline language also missed this
one. The preset's `information` category (described as "general
information, pricing or how-to") may simply not cover feature-availability
questions well in any language — consistent with the identical finding in
`recipes/support_triage/`'s "do you support SSO" miss.

**What held up perfectly:** the three unambiguous cases (an explicit refund
ask, a clear bug report, an explicit cancellation request) were classified
correctly and consistently across all three languages, with high confidence
in every case. The failures cluster specifically around content with weak
or absent intent signal (a compliment with no actual request, a feature
question that doesn't map cleanly to a category) — the model isn't
generally unreliable in Hindi/Marathi, it's unreliable exactly where the
message is ambiguous *and* non-English, which compounds two separate
weaknesses from earlier recipes (ambiguous-category misses in
`support_triage`, language bias in `phishing_detection`) into one case.

## When not to use Laya for this

- Don't assume a positive or neutral message (thanks, feedback, small talk)
  is "probably safe to leave as `other`" in non-English triage without
  testing it — this recipe found total-confidence failures on exactly that
  category, which is otherwise the lowest-stakes case in English.
- Don't trust `triage_questions()`'s `information` category in any
  language without your own validation — it missed a plain feature
  question in English *and* Hindi, and barely got it in Marathi (0.343
  confidence). This is a preset-wording problem (see `support_triage/`),
  not purely a multilingual one, but multilingual data makes it easier to
  spot since you get 3 independent reads of the same content for free.
- If you serve Hindi/Marathi support traffic at real volume, don't deploy
  on this preset's categories unmodified — build a small labeled set in
  each language you serve (even 20-30 examples per category) and check for
  exactly this kind of high-confidence, total miscategorization before
  trusting automated triage.
