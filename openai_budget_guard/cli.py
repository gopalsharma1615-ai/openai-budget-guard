"""obg — manage OpenAI key budgets, see usage, calculate costs, run the proxy.

  python -m openai_budget_guard set --key-env OPENAI_API_KEY --max-cost 5 --period monthly
  python -m openai_budget_guard set --key-env OPENAI_API_KEY --task summarize --max-output 50000
  python -m openai_budget_guard status
  python -m openai_budget_guard cost --model gpt-4o --input 12000 --cached 8000 --output 900
  python -m openai_budget_guard report --by task model
  python -m openai_budget_guard proxy --port 8787
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time

from .guard import AGGREGATE, BudgetGuard, key_id
from .store import PERIODS, window_start


def _key(args, required: bool = True) -> str | None:
    if getattr(args, "key", None):
        return args.key
    if getattr(args, "key_env", None):
        val = os.environ.get(args.key_env)
        if not val:
            sys.exit(f"environment variable {args.key_env} is empty")
        return val
    if required:
        return getpass.getpass("OpenAI API key: ").strip()
    return None


def _table(rows: list[list], header: list[str], left: int = 2) -> str:
    """Plain-text table; the first `left` columns are left-aligned, the rest right-aligned."""
    cells = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    line = lambda r: "  ".join(c.rjust(w) if i >= left else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
    return "\n".join([line(cells[0]), "  ".join("-" * w for w in widths)] + [line(r) for r in cells[1:]])


def _num(dim: str, v) -> str:
    return f"${v:,.4f}" if dim == "cost_usd" else f"{int(v):,}"


def cmd_set(g: BudgetGuard, a) -> None:
    limits = {"max_cost_usd": a.max_cost, "max_total_tokens": a.max_tokens, "max_input_tokens": a.max_input,
                  "max_cached_tokens": a.max_cached, "max_output_tokens": a.max_output}
    if all(v is None for v in limits.values()):
        sys.exit("give at least one limit: --max-cost / --max-tokens / --max-input / --max-cached / --max-output")
    key = _key(a)
    g.set_budget(key, a.task, period=a.period, label=a.label, **limits)
    scope = "all tasks combined" if a.task == AGGREGATE else f"task '{a.task}'"
    print(f"budget set for key {key_id(key)} ({scope}, period={a.period}): "
          + ", ".join(f"{k}={v}" for k, v in limits.items() if v is not None))


def cmd_remove(g: BudgetGuard, a) -> None:
    print("removed" if g.remove_budget(_key(a), a.task) else "no such budget")


def cmd_reset(g: BudgetGuard, a) -> None:
    print(f"deleted {g.reset_usage(_key(a), a.task)} ledger rows")


def cmd_status(g: BudgetGuard, a) -> None:
    key = _key(a, required=False)
    st = g.status(key)
    if a.json:
        print(json.dumps(st, indent=2))
        return
    if not st:
        print("no budgets set")
        return
    rows = []
    for s in st:
        who = s["label"] or s["key"]
        task = "* (all)" if s["task"] == AGGREGATE else s["task"]
        for dim, x in s["limits"].items():
            flag = "EXHAUSTED" if x["exhausted"] else ("low" if x["pct_used"] >= 80 else "ok")
            rows.append([who, task, s["period"], dim, _num(dim, x["used"]), _num(dim, x["limit"]),
                         _num(dim, x["remaining"]), f"{x['pct_used']:.1f}%", flag])
    print(_table(rows, ["key", "task", "period", "metric", "used", "limit", "remaining", "used%", "state"]))


def cmd_cost(g: BudgetGuard, a) -> None:
    c = g.cost(a.model, a.input, a.cached, a.output)
    p = g.prices.lookup(a.model)
    total_calls = a.calls
    rows = [["input (uncached)", f"{a.input:,}", f"${p.input:.4f}", f"${c.input_usd:.6f}", f"${c.input_usd * total_calls:,.4f}"],
            ["cached input", f"{a.cached:,}", f"${p.cached_input:.4f}", f"${c.cached_usd:.6f}", f"${c.cached_usd * total_calls:,.4f}"],
            ["output", f"{a.output:,}", f"${p.output:.4f}", f"${c.output_usd:.6f}", f"${c.output_usd * total_calls:,.4f}"],
            ["total", f"{a.input + a.cached + a.output:,}", "", f"${c.total_usd:.6f}", f"${c.total_usd * total_calls:,.4f}"]]
    print(f"model {a.model}  (prices from {g.prices.source})")
    print(_table(rows, ["part", "tokens", "$/1M", "per call", f"x {total_calls:,} calls"], left=1))


def cmd_prices(g: BudgetGuard, a) -> None:
    rows = [[m, f"{p.input:.4f}", f"{p.cached_input:.4f}", f"{p.output:.4f}"] for m, p in sorted(g.prices.prices.items())]
    print(f"USD per 1M tokens, from {g.prices.source}")
    print(_table(rows, ["model", "input", "cached", "output"], left=1))


def cmd_report(g: BudgetGuard, a) -> None:
    key = _key(a, required=False)
    since = window_start(a.period) if a.period else 0.0
    rows = g.store.report(key_id(key) if key else None, since, tuple(["key_id"] + [b for b in a.by if b != "key_id"]))
    if a.json:
        print(json.dumps(rows, indent=2))
        return
    out = []
    for r in rows:
        out.append([r["label"] or r["hint"], *[r[b] for b in a.by if b != "key_id"], r["requests"],
                    f"{r['input_tokens']:,}", f"{r['cached_tokens']:,}", f"{r['output_tokens']:,}",
                    f"${r['input_cost_usd']:.4f}", f"${r['cached_cost_usd']:.4f}", f"${r['output_cost_usd']:.4f}",
                    f"${r['cost_usd']:.4f}"])
    grp = [b for b in a.by if b != "key_id"]
    print(_table(out, ["key", *grp, "reqs", "input", "cached", "output", "$input", "$cached", "$output", "$total"])
          if out else "no usage recorded")
    if out:
        print(f"\ntotal: ${sum(r['cost_usd'] for r in rows):.4f} over {sum(r['requests'] for r in rows)} requests")


def cmd_log(g: BudgetGuard, a) -> None:
    rows = [[time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])), r["hint"], r["task"], r["model"],
             r["endpoint"], f"{r['input_tokens']:,}", f"{r['cached_tokens']:,}", f"{r['output_tokens']:,}",
             f"${r['cost_usd']:.6f}", r["status"]] for r in g.store.recent(a.limit)]
    print(_table(rows, ["time", "key", "task", "model", "endpoint", "input", "cached", "output", "cost", "status"])
          if rows else "no usage recorded")


def cmd_proxy(g: BudgetGuard, a) -> None:
    from .proxy import serve

    serve(a.host, a.port, guard=g, upstream=a.upstream, default_task=a.task)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="openai_budget_guard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="ledger path (default $OBG_DB or ~/.openai_budget_guard/usage.db)")
    p.add_argument("--pricing", help="price file (default $OBG_PRICING or bundled pricing.json)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_key(sp, task_default=AGGREGATE):
        k = sp.add_mutually_exclusive_group()
        k.add_argument("--key", help="OpenAI key (prefer --key-env to keep it out of shell history)")
        k.add_argument("--key-env", help="read the key from this environment variable, e.g. OPENAI_API_KEY")
        if task_default is not None:
            sp.add_argument("--task", default=task_default, help=f"task name; '*' = all tasks combined (default {task_default!r})")
        return sp

    s = with_key(sub.add_parser("set", help="allot / update a budget"))
    s.add_argument("--max-cost", type=float, help="USD")
    s.add_argument("--max-tokens", type=int, help="input + cached + output tokens")
    s.add_argument("--max-input", type=int, help="uncached input tokens")
    s.add_argument("--max-cached", type=int, help="cached input tokens")
    s.add_argument("--max-output", type=int, help="output tokens")
    s.add_argument("--period", choices=PERIODS, default="none", help="rolling reset window (UTC)")
    s.add_argument("--label", help="friendly name shown in status/report")
    s.set_defaults(fn=cmd_set)

    with_key(sub.add_parser("remove", help="delete a budget")).set_defaults(fn=cmd_remove)
    r = with_key(sub.add_parser("reset", help="clear recorded usage (all tasks unless --task)"), task_default=None)
    r.add_argument("--task")
    r.set_defaults(fn=cmd_reset)

    st = with_key(sub.add_parser("status", help="used / limit / remaining per budget"), task_default=None)
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_status)

    c = sub.add_parser("cost", help="cost calculator")
    c.add_argument("--model", required=True)
    c.add_argument("--input", type=int, default=0, help="uncached input tokens")
    c.add_argument("--cached", type=int, default=0, help="cached input tokens")
    c.add_argument("--output", type=int, default=0, help="output tokens")
    c.add_argument("--calls", type=int, default=1, help="multiply by this many calls")
    c.set_defaults(fn=cmd_cost)

    sub.add_parser("prices", help="list the price book").set_defaults(fn=cmd_prices)

    rp = with_key(sub.add_parser("report", help="usage + cost breakdown"), task_default=None)
    rp.add_argument("--by", nargs="+", default=["task", "model"], choices=["task", "model", "endpoint", "status"])
    rp.add_argument("--period", choices=[x for x in PERIODS if x != "none"], help="only the current window")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_report)

    lg = sub.add_parser("log", help="recent requests")
    lg.add_argument("--limit", type=int, default=20)
    lg.set_defaults(fn=cmd_log)

    px = sub.add_parser("proxy", help="run the OpenAI-compatible budget proxy")
    px.add_argument("--host", default="127.0.0.1")
    px.add_argument("--port", type=int, default=8787)
    px.add_argument("--upstream", default="https://api.openai.com/v1")
    px.add_argument("--task", default="default", help="task when a request has no X-Budget-Task header")
    px.set_defaults(fn=cmd_proxy)

    a = p.parse_args(argv)
    from .pricing import PriceBook
    g = BudgetGuard(a.db, PriceBook.load(a.pricing))
    a.fn(g, a)
    return 0
