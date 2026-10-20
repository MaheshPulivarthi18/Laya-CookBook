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

Support triage recipe is next.

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

## License

Apache-2.0. Pinned against Laya 0.3.11 -- behavior may differ on other versions.
