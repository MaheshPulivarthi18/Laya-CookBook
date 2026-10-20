# Troubleshooting

Things that actually went wrong while building this cookbook, and the real
fix for each — not speculative advice.

## `USE_TF=0`

Set this before `import laya`, every time. Laya probes for TensorFlow on
import, and without this it can hang for minutes on some environments.
Every script in this repo sets it with `os.environ.setdefault("USE_TF", "0")`
before the `from laya import ...` line. If a script hangs right after
starting with no output, this is almost certainly why.

## Slow / stuck first load

The first call to any checkpoint downloads and builds it — a few seconds
even from local cache, longer on a cold network fetch. See
`production/speed_and_memory.py` for measured numbers and the preload
pattern (`Router(preload=True)` or `router.preload([...])`) that moves this
cost to startup instead of your first real request.

## `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon

Set this alongside `USE_TF=0`. Without it, an op the MPS backend doesn't
support yet raises instead of falling back to CPU for that op.

## Hinglish (or other Latin-script non-English) routes inconsistently

This is not a bug you can configure away — `00_basics/04_router_and_languages.py`
and `recipes/hinglish/` both found the script/stopword heuristic genuinely
inconsistent on Hinglish, sometimes landing on `english`, sometimes
`multilingual`, sometimes falling back to the default. If you have real
Hinglish (or similar code-mixed) traffic, test both `model="english"` and
`model="multilingual"` on your own data — `recipes/hinglish/` found
`english` measurably *better* for ordinary classification there (83% vs.
67%), which is the opposite of what you'd guess. Don't trust auto-routing's
current behavior to land on the better checkpoint for your traffic.

## A preset's questions seem to refer to nothing

Every preset's `instructions` names a specific state field (`prompt` for
`guard_questions()`, `post` for `moderation_questions()`, `message` for
`triage_questions()`, `request` for `router_questions()`, `body` for
`email_questions()`). Using the "wrong" key mostly doesn't break anything,
though — `00_basics/05_presets.py` found `serialize_state` just
`json.dumps`s the whole state regardless of key names, so the model reads
every field either way. The real risk is the opposite problem: you can't
scope a question to one field, so anything extra you attach to the state
(retrieved context, chat history, metadata) is also judged by every
question, whether its instructions name that field or not.

## A `ValueError: ... options exceed head_max_len=...`

You have too many `choice` options (or options with long enough text) for
the token budget. See `production/many_options.py` for three fixes
(raising `head_max_len` *and* `max_len` together, `shortlist_choice`, or a
manual two-step coarse-then-fine choice) and the honest tradeoffs of each —
the first fix pushes you past the checkpoint's trained context length, and
the second has a real failure mode of its own (`00_basics/06_hidden_features.py`
found it can silently drop the correct label).

## A `RuntimeWarning` about "invalid temperatures" on checkpoint load

Seen on both `english` and `multilingual` checkpoint loads in testing:
`this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:... -> 0.5. Treat confidence from the affected entries as
uncalibrated.` This is Laya itself telling you some of its own shipped
calibration values are out of range and were silently clamped. It's not
something you can fix client-side — see `production/calibration.py` for
how to measure actual calibration (ECE) on your own labeled data rather
than trusting the model's self-reported confidence as-is.

## A result looks confidently wrong on long text

Check `result["usage"]["input_tokens"]` against the checkpoint's limit
(512 for `english`, 1024 for `multilingual`/`typed-decisions`) before
trusting anything on text near that length. If it's at or near the limit,
your text was silently truncated — Laya raises no warning for this one,
unlike the calibration warning above. See `production/long_text.py` for a
truncation check and a chunking fallback.

## `LayaGuardrail` blocks something obviously benign

Its default `threshold=0.5` is applied to every question regardless of
type — fine for `noul` (a 0–1 probability), wrong for `score` questions
like `guard_questions()`'s `harm_severity` (a 0–3 ordinal scale, where 0.5
is an almost-zero bar). `00_basics/06_hidden_features.py` reproduced this
blocking "What is the capital of France?". Pass your own `questions` with
score-type questions excluded, or handle them with a separate, type-aware
threshold (see `production/confidence_gating.py`).

## Confidence doesn't mean what you'd expect

See `00_basics/03_reading_the_output.py` — it's a different formula per
question type (`1 - entropy/log(K)` for `choice`/`score`, `max(p, 1-p)` for
`noul`), proven by recomputing it against live output, not just asserted.
