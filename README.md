# openai-budget-guard

[![CI](https://github.com/gopalsharma1615-ai/openai-budget-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/gopalsharma1615-ai/openai-budget-guard/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue) ![openai](https://img.shields.io/badge/openai%20SDK-1.x%20%7C%202.x%20%7C%203.x-green) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

Token and cost budgets for OpenAI API keys. You give a key a budget, overall and/or per task. Each request is
checked before it is sent and metered after it returns, with input, cached input and output counted separately.
Once a budget is used up, the request never reaches OpenAI and the caller gets:

```
openai.RateLimitError: Cost budget exhausted for OpenAI key sk-...abcd, all tasks combined, monthly window: used $5.0012 of $5.0000.
openai.RateLimitError: Token budget exhausted for OpenAI key sk-...abcd, task 'summarize': used 50,183 output tokens of 50,000 output tokens.
```

`e.code` is `cost_budget_exhausted`, `token_budget_exhausted` or `no_budget_allotted`. The response is HTTP 429
with `x-should-retry: false`, so the OpenAI SDK raises the error immediately and does not retry.

## Install

```bash
pip install "git+https://github.com/gopalsharma1615-ai/openai-budget-guard"            # library + CLI (`obg`)
pip install "openai-budget-guard[proxy] @ git+https://github.com/gopalsharma1615-ai/openai-budget-guard"   # + proxy server
```

Works with openai SDK 1.x, 2.x and 3.x (3.x moved to `httpx2`; the guard picks whichever the installed SDK uses).

## Concepts

| | |
|---|---|
| **Key** | Identified by `sha256(key)[:16]`. The raw key is never stored; reports show `sk-...abcd` or your label. |
| **Task** | A free-form name per request, such as `summarize`, `rag` or `demo-1`. Set it on the client or per call. |
| **Budget** | `task="*"` covers all of the key's tasks combined. Any other task name covers only that task. A request must pass every budget that applies to it. |
| **Limits** | Any mix of `max_cost_usd`, `max_total_tokens`, `max_input_tokens` (uncached), `max_cached_tokens` and `max_output_tokens` (includes reasoning tokens). |
| **Period** | `none` (lifetime), `hourly`, `daily`, `weekly` or `monthly`. Windows are in UTC. |
| **Preflight** | Blocks a request if its worst case (prompt tokens + `max_tokens`) would overshoot the budget. Without `max_tokens`, 512 output tokens are assumed (`OBG_DEFAULT_OUTPUT_RESERVE`). The estimate is reserved atomically, so concurrent requests cannot overshoot together. After the response it is replaced by actual usage. |

## Python (any OpenAI-SDK app)

```python
from openai_budget_guard import BudgetGuard, guarded_openai, for_task

guard = BudgetGuard()                                          # ledger: $OBG_DB or ~/.openai_budget_guard/usage.db
guard.set_budget(key, "*", max_cost_usd=5, period="monthly", label="sales-demo")
guard.set_budget(key, "summarize", max_output_tokens=50_000)

client = guarded_openai(key, task="summarize", guard=guard)   # a normal openai.OpenAI
client.chat.completions.create(...)                            # chat, responses, embeddings, streaming: all metered
for_task(client, "rag").responses.create(...)                  # same client, billed to task "rag"
client.chat.completions.create(..., extra_headers={"X-Budget-Task": "qa"})   # or per call
```

Async: `guarded_async_openai(...)`. LangChain and other httpx-based clients take the transport directly:
`ChatOpenAI(http_client=openai.DefaultHttpxClient(transport=BudgetTransport(guard, "task")))`.

## Proxy (any language, no code changes)

```bash
python -m openai_budget_guard proxy --port 8787
```

Point the app at `base_url=http://localhost:8787/v1` and keep its own OpenAI key. Send the
`X-Budget-Task: <task>` header to pick a task. `GET /budget/status` with the key as Bearer returns that key's
budgets.

## CLI

```bash
obg set --key-env OPENAI_API_KEY --max-cost 5 --period monthly --label sales-demo
obg set --key-env OPENAI_API_KEY --task summarize --max-output 50000 --max-cached 1000000
obg status                         # used / limit / remaining / state for every budget
obg report --by task model         # input / cached / output tokens and $ per key, task and model
obg log                            # last requests
obg cost --model gpt-4o --input 12000 --cached 8000 --output 900 --calls 1000   # calculator
obg prices
obg reset --key-env OPENAI_API_KEY [--task summarize]
obg remove --key-env OPENAI_API_KEY --task summarize
```

## Prices

Prices live in `openai_budget_guard/pricing.json` as USD per 1M tokens. Model names are matched by longest prefix,
so `gpt-4o-2024-08-06` is priced as `gpt-4o`. **Check the numbers against openai.com/api/pricing**, then edit the
file or point `OBG_PRICING` at your own copy. A model with no price is refused with `unknown_model_pricing`. To
allow it anyway at $0 (tokens are still counted), use `BudgetGuard(unknown_model="allow")`.

## Notes / limits

- Only `chat/completions`, `responses`, `embeddings` and `completions` are metered. Other endpoints (files,
  models, ...) pass through untouched. Image, audio and realtime usage is not priced.
- For streamed chat, the guard adds `stream_options.include_usage`. If your code did not ask for the usage chunk,
  the guard removes it again. If a stream is abandoned before usage arrives, the worst-case estimate stays in the
  ledger with status `estimated`.
- Requests that fail upstream (4xx/5xx) release their reservation, because OpenAI does not bill them.
- The ledger is SQLite in WAL mode. It is safe across threads and across processes on one machine. Use one DB
  path (`OBG_DB`) for every app that should share a budget.

## Tests

`python -m pytest tests -q`: 14 tests through the real `openai` SDK against a mocked server, so no network and no
spend. `python examples/demo.py --mock` shows the block messages.
