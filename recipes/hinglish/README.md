# Recipe: Hinglish (code-mixed Hindi-English)

## Problem

`00_basics/04_router_and_languages.py` found the routing heuristic
inconsistent on Hinglish — the same style of text sometimes reads as
English, sometimes as "undecided." This recipe goes past routing and asks
the practical question: **if you have real Hinglish traffic, which
checkpoint should you actually pin it to?** 6 triage messages and 2
guard messages, all genuinely code-mixed (Hindi words transliterated into
Latin script, mixed with English), each run three ways: auto-routed,
forced onto `english`, forced onto `multilingual`.

## Question design, and why

Triage messages reuse `triage_questions()` (comparable to
`recipes/support_triage/` and `recipes/multilingual_hindi_marathi/`),
including the same "pure compliment, zero complaint vocabulary" trap that
produced a 100%-confidence failure on pure Hindi/Marathi. Guard messages
reuse `guard_questions()` with a phishing-style and a jailbreak-style
prompt, both in Hinglish, to check recall independent of which checkpoint
runs them.

## Code

[`predict.py`](predict.py) runs every message under all three model
settings (`model=None` auto, `model="english"`, `model="multilingual"`)
and reports accuracy per setting.

## Real output

```
Summary: accuracy by forced model
  auto         triage_intent_accuracy=83%  guard_recall=100%
  english      triage_intent_accuracy=83%  guard_recall=100%
  multilingual triage_intent_accuracy=67%  guard_recall=100%
```

The two misses under `multilingual` (and nowhere else):

```
compliment: "Bahut dhanyavaad, aapki support team ne kamaal kiya!"
  english      -> other   (correct, conf=0.460)
  multilingual -> refund  (WRONG, conf=0.463)      <- same failure as the pure-language recipe

information_request: "Kya is plan mein priority support milta hai?"
  english      -> other             (wrong, conf=0.372)
  multilingual -> billing_question  (wrong, conf=0.444, different wrong answer)
```

Full numbers in `output/results.json`.

## What it gets wrong

**The intuitive choice — force Hinglish onto `multilingual` since it's not
"clean" English — is measurably the worse choice here.** 83% vs. 67% on
identical content, forced model the only variable. And the failure that
caused the gap is exactly the one `recipes/multilingual_hindi_marathi/`
found on pure Hindi/Marathi: the same compliment, this time code-mixed,
again got misread as a `refund` request specifically under `multilingual` —
not as confidently (0.463 vs. 1.0 for pure Hindi), but wrong in the same
direction, on the same checkpoint, on conceptually the same content. This
looks like a real characteristic of the `multilingual` checkpoint on this
preset, not a one-off: it's biased toward reading low-content/positive
messages as requests, and that bias follows the checkpoint wherever
Hinglish gets routed onto it.

**Auto-routing mostly avoided this by accident, not by design.** In this
test, the heuristic sent 5 of 6 triage messages to `english` and only one
(`technical_crash`) to `multilingual` — not because it correctly detected
"this is English-dominant Hinglish," but because of the same
script/stopword heuristic inconsistency `04_router_and_languages.py`
already documented. **Don't rely on auto-routing's current behavior to
keep landing on the better checkpoint for your specific Hinglish traffic —
it got lucky on this set's exact wording, not on a principled read of code-
mixed text.**

**Guard recall (jailbreak/phishing detection) was unaffected by model
choice — 100% under all three settings**, consistent with
`00_basics/04_router_and_languages.py`'s finding that an injection phrase
in English-adjacent Hinglish gets caught regardless of which checkpoint
runs it. The model-choice problem here is specific to ordinary
classification (`triage_questions()`), not to safety questions.

**The `information_request` miss happened on both forced models, with
different wrong answers** — the same preset-wording gap found in
`support_triage/` and `multilingual_hindi_marathi/`, now confirmed a third
time, in a third language mode.

## When not to use Laya for this

- Don't default to forcing `multilingual` for Hinglish traffic just because
  it "isn't English." For ordinary triage/classification, this test found
  `english` measurably better (83% vs. 67%) on the same content — test
  both on your own Hinglish data before picking one, rather than assuming
  the "more multilingual" checkpoint is the safer default.
- Don't trust auto-routing's current split-decision behavior on Hinglish to
  be stable or correct for your traffic — it's driven by the same fragile
  heuristic documented in `00_basics/04_router_and_languages.py`, and this
  recipe shows it can land on the worse checkpoint for some messages and
  the better one for others, inconsistently, within the same style of text.
- For non-adversarial classification specifically, be extra wary of
  positive/neutral/low-content Hinglish messages (compliments, small talk)
  — this is the third recipe in a row where exactly that category produced
  a confident wrong answer, now confirmed across pure Hindi, pure Marathi,
  and code-mixed Hinglish alike.
