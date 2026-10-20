# Recipe: email routing

## Problem

An inbox gets emails that need to go to different teams (billing, technical,
sales, security, HR), some of which are spam or phishing, some of which are
urgent, and all of which need a decision in milliseconds if you're doing it
at the volume a real support inbox sees. You don't want an LLM call per
email, and you don't want to train a classifier from scratch.

## Question design, and why

We reuse `laya.email_questions()` rather than writing new questions, because
it already ships 5 questions tuned for exactly this: `category` (choice, 6
labels), `is_spam` (noul), `is_phishing` (noul), `urgency` (score, 3 levels),
`needs_reply` (noul) -- all in one forward pass per email. It expects the
state to have a `body` key (and tolerates extra keys like `subject` — see
`00_basics/05_presets.py` on why extra fields don't need to be scoped out,
they're just also read).

The test set (`data/emails.json`, 10 emails) isn't just the easy cases. It
includes one genuinely ambiguous email, one real-but-informal marketing
email (to test spam vs. legitimate correspondence), and one non-English
(Hindi) complaint, specifically to find where routing breaks, not just where
it works.

## Code

[`predict.py`](predict.py) loads the 10 emails, runs `email_questions()` on
each through `Router(device="mps")`, scores category accuracy against a
hand-labeled expectation, and writes every raw answer to `output/results.json`.

## Real output

```
[OK  ] billing_duplicate_charge         expected=billing    predicted=billing    conf=0.990
[OK  ] technical_outage                 expected=technical  predicted=technical  conf=0.796
[OK  ] sales_pricing                    expected=sales      predicted=sales      conf=0.851
[OK  ] security_phishing                expected=security   predicted=security   conf=0.961  spam=0.93 phish=0.96
[OK  ] hr_leave_request                 expected=hr         predicted=hr         conf=0.498
[MISS] other_newsletter                 expected=other      predicted=technical  conf=0.261
[OK  ] ambiguous_billing_or_technical   expected=billing    predicted=billing    conf=0.965
[OK  ] urgent_legit_refund              expected=billing    predicted=billing    conf=0.863
[MISS] legit_marketing_not_spam         expected=sales      predicted=other      conf=0.160
[OK  ] hindi_billing_complaint          expected=billing    predicted=billing    conf=0.782  spam=0.95 phish=0.97  <- see below

Category accuracy: 8/10 = 80%
```

Full per-email probabilities, confidences, and token counts are in
`output/results.json` (a real run, not a hand-edited sample).

## What it gets wrong

**Two low-confidence category misses, and the confidence correctly flags
both of them.** `other_newsletter` (a design-blog roundup) landed on
`technical` (47.9%) over `other` (27.7%) — "Figma plugins" and "component
variants" read as tech content to the model. `legit_marketing_not_spam` (a
customer-success follow-up with a usage report attached) split almost evenly
across billing/technical/sales/other and landed on `other`. Neither miss is
a confident, silent failure — both carry confidence under 0.27, correctly
signaling "don't trust this one." The deeper issue is that neither email
cleanly fits any of the 6 fixed categories; a relationship-based
customer-success email isn't really "sales" or "other." **If your real
inbox has a category like that, add it to `criteria` rather than forcing
the preset's 6 labels.**

**The serious one: a legitimate Hindi billing complaint got flagged
`is_spam=0.95` and `is_phishing=0.97`.** The `category` call got this email
right (billing, 90.7%) — the failure is specific to the two `noul` safety
questions on non-English input. In a real pipeline, a legitimate non-English
customer complaint would get auto-flagged as phishing and likely buried or
blocked. This is measured on one example, not a benchmark — but it's exactly
where you'd want a `laya-bench`-style evaluation on a real labeled
multilingual spam/phishing set before trusting these two questions on
non-English mail in production. It may be connected to the calibration
warning Laya itself raises on loading the multilingual checkpoint (`this
checkpoint ships invalid temperatures... treat confidence from the affected
entries as uncalibrated` — see the root README) but the *value*, not just the
confidence, looks wrong here, so don't assume recalibration alone fixes it.

## When not to use Laya for this

- If your inbox needs more than ~6-10 genuinely distinct categories with
  fine-grained boundaries (e.g. 30 product-specific queues), check
  `production/many_options.py` (shortlisting) first — and note the cookbook's
  own `06_hidden_features.py` found the default shortlist embedder can drop
  the correct label entirely. Don't trust it unverified at that scale.
- If you need to act automatically on `is_spam`/`is_phishing` for non-English
  mail at any real volume, this recipe's Hindi result says: not yet, not
  without your own evaluation data. Route non-English phishing/spam
  decisions to a human reviewer, or re-verify on a labeled set, before
  automating on them.
- If "category" genuinely requires understanding content beyond a page of
  text (e.g. a long forwarded thread with the real request buried past token
  512/1024), check `usage.input_tokens` in the output — truncation is silent.
