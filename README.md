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
| `04_router_and_languages.py` | *(next)* how routing picks a checkpoint, and where the heuristic gets it wrong |
| `05_presets.py` | *(next)* every built-in preset, the state key each one expects |
| `06_hidden_features.py` | *(next)* `LayaGuardrail`, hooks, `shortlist_choice`, the HTTP server, the MCP server |

### `recipes/` and `production/`
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

## License

Apache-2.0. Pinned against Laya 0.3.11 -- behavior may differ on other versions.
