# Recipe: phishing detection

## Problem

`recipes/email_routing/` found one alarming data point: a legitimate Hindi
billing complaint got flagged `is_phishing=0.97`. One example isn't evidence
of a pattern, so this recipe tests properly: 16 messages, hand-labeled
phishing/legitimate, in English, Hindi and Marathi, with matched content
where possible (the same billing complaint appears in both English and
Hindi, almost word for word) so language is isolated from content as a
variable.

## Question design, and why

We run **two** versions of `is_phishing` on every message:

- `default_q` — exactly `email_questions()`'s criteria (`presets.py`):
  terse, two-line true/false descriptions.
- `detailed_q` — a longer, more specific rewrite, naming concrete tells
  (asks for a link click / credentials / card details under urgency) vs.
  what a routine notice looks like.

This directly tests your Step 0 finding that "wording matters" (a custom
criteria shifted a legitimate email's phishing probability from 0.19 to
0.38) — does a more careful rewrite fix the cross-lingual gap, or not?

## Code

[`predict.py`](predict.py) runs both question variants over all 16 messages
through `Router(device="mps")`, buckets accuracy by script (Latin vs.
Devanagari), and lists every false positive explicitly.

## Real output

```
DEFAULT criteria                     DETAILED criteria
Overall accuracy: 75%                Overall accuracy: 75%
  latin     : 9/11 = 82%               latin     : 9/11 = 82%
  devanagari: 3/5  = 60%               devanagari: 3/5  = 60%

False positives (identical in both):
  legit_real_password_reset      (latin,      noul 0.65 / 0.58)
  legit_urgent_but_real          (latin,      noul 0.76 / 0.69)
  legit_hindi_billing_complaint  (devanagari, noul 0.65 / 0.91)  <- WORSE with detailed wording
  legit_hindi_password_reset    (devanagari, noul 0.89 / 0.95)  <- WORSE with detailed wording
```

Full per-message numbers for both variants are in `output/results.json`.

## What it gets wrong

**Finding 1 — the two Latin-script false positives are a real, defensible
limitation, not an obvious bug.** `legit_real_password_reset` ("We received
a request to reset your password...") and `legit_urgent_but_real` ("your
free trial ends tomorrow...add a payment method...before midnight") both
read structurally like phishing templates, because real phishing *mimics*
exactly this kind of transactional urgency. Both land at moderate-not-extreme
confidence (0.58–0.76), which is arguably honest uncertainty rather than a
confident miss. If you're building a guardrail on `is_phishing`, expect this
category of false positive and plan for it (e.g. a second check: does the
message actually request credentials/payment, vs. merely mention a password
or deadline).

**Finding 2 — the Hindi bias is real and isolated from content, not just
language-adjacent.** `legit_billing_complaint` (English: "We were billed
twice for March. Please refund the duplicate today or we will cancel our
plan.") scored `noul=0.17` — correctly, confidently legitimate. The nearly
word-for-word Hindi translation of the *same complaint*, `legit_hindi_billing_complaint`,
scored `noul=0.65` (default) and `0.91` (detailed) — wrong both times, and
*more* wrong with more careful wording. Content held constant, language
changed, the answer flipped. That rules out "the Hindi examples just
happened to sound more suspicious" as an explanation.

**Finding 3 — rewriting the criteria did not fix it, and made the Hindi
case worse.** Both question variants produced the *exact same* 4 false
positives, at the exact same accuracy (75% overall, 60% on Devanagari
script). For the two Hindi legitimate examples, the more detailed wording
*increased* the wrongly-high phishing probability (0.65→0.91, 0.89→0.95).
So: **wording changes magnitude, it does not fix a systematic bias.** If you
only have one language's examples to validate on, don't assume your
prompt-engineering generalizes to others — test on the actual language you
plan to deploy to.

**Marathi, for what it's worth, was perfect (2/2)** — both the legitimate
invoice notice and the phishing message classified correctly. With only one
example each, treat this as "not conclusively broken for Marathi," not "Hindi's
problem doesn't apply elsewhere."

## When not to use Laya for this

- Don't auto-block on `is_phishing >= 0.5` for non-English traffic without
  your own labeled evaluation set in that language first — this recipe
  measured 60% accuracy on 5 Devanagari-script examples, well below the
  82% on Latin script, with real customer complaints getting flagged.
- Don't assume rewriting the question's criteria fixes a cross-lingual
  accuracy gap — it didn't here, and made one case worse. If you need
  better non-English phishing detection, that likely needs calibration data
  in that language (see `laya-studio`'s planned test bench), not prompt
  tweaks.
- For transactional-but-urgent legitimate mail (password resets, trial
  expiry notices), expect moderate false-positive risk at any wording. Pair
  `is_phishing` with a second, narrower check (e.g. "does this literally ask
  for a password or card number") before auto-blocking.
