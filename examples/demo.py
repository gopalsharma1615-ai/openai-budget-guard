"""Demo: give a key a small budget, run two tasks, watch the budget message appear.

    export OPENAI_API_KEY=sk-...
    python examples/demo.py            # real calls (spends a few cents at most)
    python examples/demo.py --mock     # no network, fake OpenAI
"""
import sys
from pathlib import Path

import openai

from openai_budget_guard._http import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai_budget_guard import BudgetGuard, for_task, guarded_openai

mock = "--mock" in sys.argv
inner = None
key = None
if mock:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_guard import FakeOpenAI
    inner, key = httpx.MockTransport(FakeOpenAI()), "sk-mock-0000000000000000demo"

guard = BudgetGuard("demo_usage.db")
client = guarded_openai(key, task="summarize", guard=guard, inner=inner)
key = client.api_key
guard.reset_usage(key)

guard.set_budget(key, "*", max_cost_usd=0.01, label="demo")          # aggregate: 1 cent across all tasks
guard.set_budget(key, "summarize", max_output_tokens=400)             # individual: summarize may emit 400 tokens
guard.set_budget(key, "qa", max_input_tokens=2_000)                   # individual: qa may read 2k uncached tokens

qa = for_task(client, "qa")
for i in range(6):
    for name, c in (("summarize", client), ("qa", qa)):
        try:
            r = c.chat.completions.create(model="gpt-4o-mini", max_tokens=150,
                                          messages=[{"role": "user", "content": f"One line about the number {i}."}])
            print(f"[{name}] ok: {r.choices[0].message.content[:60]!r}")
        except openai.RateLimitError as e:
            print(f"[{name}] BLOCKED ({e.code}): {e.body['message'] if isinstance(e.body, dict) else e}")

for s in guard.status(key):
    for dim, x in s["limits"].items():
        used = f"${x['used']:.4f}" if dim == "cost_usd" else f"{int(x['used']):,}"
        print(f"{s['task']:>10} {dim:>14}: {used} / {x['limit']:,} ({x['pct_used']}%)")
