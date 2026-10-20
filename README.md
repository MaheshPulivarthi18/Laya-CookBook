# laya-cookbook

Runnable recipes for [Laya](https://github.com/NandhaKishorM/laya), the open-source
"System One" decision model: you give it a state (some text) and typed questions
(`choice` / `score` / `noul`), and it answers every question in one fast, local,
non-autoregressive forward pass -- probabilities in, probabilities out, no text
generation, no parsing, no chance of an out-of-schema answer.

This repo answers ["how do I actually use Laya?"](https://github.com/NandhaKishorM/laya/issues/118)
with working code and real output, not just API docs.

Part of a four-repo series:
1. **laya-cookbook** (this repo) -- how do I use Laya?
2. `laya-studio` -- what happens when I change this? *(planned)*
3. `signalbox` -- what can I build with it? *(planned)*
4. `laya-bench` -- should I use Laya or an LLM? *(planned)*

## 5-minute quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install laya

export USE_TF=0                        # avoids a known TF-loading hang
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Apple Silicon

python 00_basics/01_first_prediction.py
```

```python
from laya import Router

result = Router().predict(
    {"body": "Hi, we were billed twice for March. Please refund the duplicate today."},
    {"department": {"type": "choice",
        "instructions": "Which department should handle the request in `body`?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts", "other": "everything else"}}},
)
print(result["answers"]["department"]["choice"])   # -> "billing"
```

## Recipe index

### `00_basics/`
| Recipe | What it teaches |
|---|---|
| [`01_first_prediction.py`](00_basics/01_first_prediction.py) | One state, one question, every line explained |
| [`02_three_question_types.py`](00_basics/02_three_question_types.py) | `choice` vs `score` vs `noul`, side by side on the same input |
| [`03_reading_the_output.py`](00_basics/03_reading_the_output.py) | Every output field, and the confidence formulas *proven by recomputing them* |
| [`04_router_and_languages.py`](00_basics/04_router_and_languages.py) | How routing picks a checkpoint, and where the heuristic gets it wrong (Hinglish) |
| [`05_presets.py`](00_basics/05_presets.py) | All 5 presets, the state key each expects, and why keys don't scope a question |
| [`06_hidden_features.py`](00_basics/06_hidden_features.py) | `LayaGuardrail` (and a real default-config bug in it), hooks, `shortlist_choice` (and a real shortlisting failure), `laya.serve`/`laya.mcp` by reference |

`00_basics/` is complete. Week-1 target also includes the first 3 `recipes/`.

### `recipes/`
| Recipe | What it covers |
|---|---|
| [`email_routing/`](recipes/email_routing/) | Category + spam/phishing/urgency on 10 real emails (80% category accuracy) — surfaces a serious false-positive: a legitimate Hindi complaint flagged `is_phishing=0.97` |
| [`phishing_detection/`](recipes/phishing_detection/) | 16 messages across English/Hindi/Marathi, 2 criteria wordings — isolates a real language-driven false-positive bias (same complaint: 0.17 English vs. 0.65–0.91 Hindi) that rewriting the question does not fix |
| [`support_triage/`](recipes/support_triage/) | 13 tickets through `triage_questions()` (85% intent accuracy) — the more important miss is an *easy*, unambiguous question losing confidently (70%) to `other`, not the hard edge case |
| [`jailbreak_guard/`](recipes/jailbreak_guard/) | 14 prompts testing both directions (real jailbreaks vs. benign-but-similar text) — a complete miss on hypothetical-framing jailbreaks (0.001), a false positive on ordinary "ignore the typo" phrasing, and a tested AND-rule fix that trades the false positive for a new false negative |
| [`llm_routing/`](recipes/llm_routing/) | 16 requests through `router_questions()` — the top `difficulty` level never once reached "hard" even for genuinely hard requests (use `score`, not argmax), and a medical-safety question scored low on both `domain` and `is_sensitive`, the two signals meant to catch it |
| [`fact_checking/`](recipes/fact_checking/) | First real use of `LayaEvaluator` for hallucination detection — every false positive is an entity-swap (wrong name/date/number in an otherwise-faithful sentence); a verified threshold sweep shows accuracy jumps from 75% to 100% by raising the cutoff from 0.5 to 0.7 |
| [`multilingual_hindi_marathi/`](recipes/multilingual_hindi_marathi/) | Same 5 messages translated into English/Hindi/Marathi through `triage_questions()` — a pure compliment ("thank you, your team was amazing") was read as a `refund` request at 100% confidence in both Hindi and Marathi |

Week 1 (basics + 3 recipes) is complete. 4 of 6 remaining week-2 recipes done
(hinglish, conversation_scoring still to come).

### `production/`
Not yet written -- see the project roadmap.

## Environment this was verified on

Laya **0.3.11**, Python 3.12, macOS arm64 (Apple M4, MPS backend), torch 2.14.0.
Every number printed by these scripts is a real run, not a doc claim.

## Honest limits (found while writing this)

- **Silent truncation.** Text beyond the checkpoint's context limit (512 tokens
  english / 1024 multilingual, typed-decisions) is silently dropped -- no warning,
  and the model can be confidently wrong as a result. Always check `usage.input_tokens`.
- **`action.act_probability` is inert** in 0.3.11 -- always `1.0` regardless of
  confidence. Don't build decisions on it.
- **Confidence is not comparable across question types**, and for `choice`/`score`
  it also shrinks as the option/level count grows. See `03_reading_the_output.py`.
- **The multilingual checkpoint has shipped a calibration warning** in this version
  (`invalid temperatures... treat confidence from the affected entries as
  uncalibrated`) -- seen on both `english` and `multilingual` loads in testing.
- **Language routing is a script + stopword heuristic, not a language-ID model.**
  It's reliable for script (Devanagari/Tamil/etc. always route correctly) but
  fuzzy for Latin-script languages close to English, including Hinglish.
- **State keys are hints, not bindings — you cannot scope a question to one
  field.** `serialize_state` is `json.dumps(state)`, so the model sees the whole
  state. A guard question that names `` `prompt` `` fires just as hard on an
  injection placed in a *different* field (measured: 0.9999 vs 0.9998). Keep
  states minimal — anything you attach as background context gets judged too.
- **`LayaGuardrail`'s default threshold has a real bug.** It compares every
  question against one shared scalar (default 0.5) regardless of type. That's
  fine for `noul` (a 0–1 probability) but wrong for `score` (`harm_severity` is
  0–3): a completely benign question ("What is the capital of France?") was
  blocked, because its score (~0.69) cleared 0.5 even though the model
  correctly judged it as near-zero harm. Verified against 0.3.11 source —
  don't use the defaults unmodified.
- **The fallback shortlist embedder can silently drop the correct answer.**
  `embed_fn_from_agent` mean-pools Laya's own encoder for similarity, and in
  one real test it dropped `billing` from the shortlist on an obviously
  billing-related message, producing a wrong final answer. The package's own
  docstring says a dedicated bi-encoder "will usually shortlist better" — this
  is why. Don't trust the fallback embedder unverified in production.
- **A legitimate non-English email was flagged as phishing at 97% confidence
  — and this generalizes, it isn't a one-off.** `recipes/phishing_detection/`
  isolated it with a matched pair: the identical billing complaint scores
  `is_phishing=0.17` in English (correct) vs. `0.65–0.91` in Hindi (wrong),
  content held constant. Devanagari-script accuracy measured 60% vs. 82% on
  Latin script (n=16). Rewriting the question's criteria did not fix this
  and made the Hindi case worse. Don't auto-block on `is_phishing` for
  non-English mail without your own evaluation data.
- **A plain, easy support question lost to `other` more confidently than a
  genuinely ambiguous one.** `recipes/support_triage/`: "do you support SSO
  login" scored `other=70%` vs. `information=12%` under `triage_questions()`
  — worse than the known-hard "payment failed twice" case (62% vs. 22%).
  Preset category descriptions can be narrower than your actual traffic;
  test against real phrasing, not just the textbook example per category.
- **Hypothetical-framing jailbreaks are a complete blind spot in one test.**
  `recipes/jailbreak_guard/`: "Hypothetically, if you had no content
  policy..." scored `jailbreak=0.001` — not borderline, a total miss — while
  direct-override, persona-assignment and multi-step jailbreaks were all
  caught cleanly (1.000). Meanwhile an ordinary "ignore the typo in my
  previous message" scored `jailbreak=0.955`, a false positive from
  surface lexical overlap. An AND-rule (require `jailbreak` and
  `prompt_injection` both ≥0.5) fixes the false positive but creates a new
  false negative on a real jailbreak with weaker injection signal
  (0.447) — there's no simple threshold fix here; test any combination
  rule against adversarial examples before trusting it.
- **`difficulty`'s top probability level almost never reaches "hard," even
  for genuinely hard requests.** `recipes/llm_routing/`: three clearly hard
  requests (a distributed rate limiter design, a math proof, a disguised
  multi-step word problem) all topped out at "moderate" as the single most
  likely level, despite scoring 1.73–2.17 on the continuous `score`. Gate
  on `score`, not `argmax(probabilities)`, for the same reason
  `03_reading_the_output.py` warns about `score` generally. Separately, a
  medical-safety question scored low on both `domain` (landed in
  `chitchat`) and `is_sensitive` (0.07) — the two signals meant to flag it
  for careful handling both missed.
- **Hallucination checking (`LayaEvaluator`) reliably catches *added*
  content but not *swapped* entities.** `recipes/fact_checking/`: every
  false positive (a fabricated claim marked "supported") was a single
  swapped name, date, or number inside an otherwise-faithful sentence,
  scoring 0.64–0.65 — just over the default 0.5 threshold, while genuinely
  supported claims scored ≥0.90. A verified threshold sweep shows raising
  the cutoff to ≥0.7 fixes all 3 misses on this test set (75%→100%), but
  don't trust this check alone for high-stakes numbers/names/dates even
  after recalibrating.
- **A pure compliment was read as a refund request at 100% confidence, in
  two different languages.** `recipes/multilingual_hindi_marathi/`: "Thank
  you so much, your support team was amazing!" — zero billing/refund
  vocabulary in any language — scored `refund=1.0000` in Hindi and
  `refund=0.9999` in Marathi, while English correctly read it as `other`.
  Not a borderline miscalibration: total certainty assigned to the
  opposite of the correct answer, in both non-English languages tested.

## License

Apache-2.0. Pinned against Laya 0.3.11 -- behavior may differ on other versions.
