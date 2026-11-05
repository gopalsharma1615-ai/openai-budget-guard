"""BudgetGuard: per-key / per-task budgets on cost and on input, cached and output tokens."""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from .pricing import Cost, PriceBook, PricingError
from .store import AGGREGATE, LIMITS, UsageStore, window_start

log = logging.getLogger("openai_budget_guard")

TASK_HEADER = "X-Budget-Task"
DEFAULT_TASK = "default"

_DIM_LABEL = {"total_tokens": "tokens", "input_tokens": "input tokens", "cached_tokens": "cached input tokens",
              "output_tokens": "output tokens"}
_SKIP_KEYS = {"image_url", "url", "file_data", "file_id", "data", "b64_json", "role", "type", "detail"}


def key_id(api_key: str) -> str:
    """Stable id for a key; the raw key is never stored."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def key_hint(api_key: str) -> str:
    return f"{api_key[:3]}...{api_key[-4:]}" if len(api_key) > 10 else "***"


@dataclass
class Usage:
    input_tokens: int = 0     # uncached input
    cached_tokens: int = 0    # cached input
    output_tokens: int = 0    # output incl. reasoning

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens + self.output_tokens


@dataclass
class Reservation:
    row_id: int
    key_id: str
    task: str
    model: str
    endpoint: str
    estimate: Usage


class BudgetExceeded(Exception):
    """The request was refused because a token or cost budget is (or would be) exhausted."""

    def __init__(self, message: str, violation: dict, key_hint: str, task: str):
        super().__init__(message)
        self.message = message
        self.violation = violation
        self.key_hint = key_hint
        self.task = task

    @property
    def code(self) -> str:
        dim = self.violation.get("dimension")
        if dim is None:
            return "no_budget_allotted"
        return "cost_budget_exhausted" if dim == "cost_usd" else "token_budget_exhausted"

    def to_openai_error(self) -> dict:
        """Body shaped like an OpenAI error so every OpenAI SDK surfaces `message` and `code`."""
        return {"error": {"message": self.message, "type": "budget_exceeded", "code": self.code, "param": None,
                          "budget": {"key": self.key_hint, "task": self.task, **self.violation}}}


def _fmt(dim: str, value: float) -> str:
    return f"${value:,.4f}" if dim == "cost_usd" else f"{int(value):,} {_DIM_LABEL[dim]}"


def _message(v: dict, hint: str) -> str:
    dim = v["dimension"]
    if dim is None:
        return f"No budget allotted for OpenAI key {hint}; requests are refused until one is set."
    kind = "Cost" if dim == "cost_usd" else "Token"
    scope = "all tasks combined" if v["budget_task"] == AGGREGATE else f"task '{v['budget_task']}'"
    period = "" if v["period"] == "none" else f", {v['period']} window"
    head = f"OpenAI key {hint}, {scope}{period}"
    if v["exhausted"]:
        return f"{kind} budget exhausted for {head}: used {_fmt(dim, v['used'])} of {_fmt(dim, v['limit'])}."
    left = max(v["limit"] - v["used"], 0)
    return (f"{kind} budget insufficient for {head}: this request may need up to {_fmt(dim, v['requested'])} "
            f"but only {_fmt(dim, left)} of {_fmt(dim, v['limit'])} remains.")


@lru_cache(maxsize=16)
def _encoder(model: str):
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _count_tokens(model: str, text: str) -> int:
    enc = _encoder(model)
    return len(enc.encode(text, disallowed_special=())) if enc else len(text) // 4 + 1


def _collect_text(obj, out: list[str]) -> None:
    if isinstance(obj, str):
        if not obj.startswith("data:"):
            out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in _SKIP_KEYS:
                _collect_text(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_text(v, out)


class BudgetGuard:
    """Checks budgets before a request, records actual usage after it.

    preflight=True also refuses a request whose *worst-case* estimate (prompt tokens + max output tokens)
    would overshoot the remaining budget; with False only already-exhausted budgets block.
    """

    def __init__(self, db_path: str | None = None, pricebook: PriceBook | None = None, *,
                 preflight: bool = True, default_output_reserve: int = 512,
                 unknown_model: str = "block", require_budget: bool = False):
        if unknown_model not in ("block", "allow"):
            raise ValueError("unknown_model must be 'block' or 'allow'")
        self.store = UsageStore(db_path)
        self.prices = pricebook or PriceBook.load()
        self.preflight = preflight
        self.default_output_reserve = int(os.environ.get("OBG_DEFAULT_OUTPUT_RESERVE", default_output_reserve))
        self.unknown_model = unknown_model
        self.require_budget = require_budget

    # ---- budget administration -----------------------------------------
    def set_budget(self, api_key: str, task: str = AGGREGATE, *, max_cost_usd: float | None = None,
                   max_total_tokens: int | None = None, max_input_tokens: int | None = None,
                   max_cached_tokens: int | None = None, max_output_tokens: int | None = None,
                   period: str = "none", label: str | None = None) -> None:
        """Allot a budget. task='*' caps all tasks of the key combined; any other task caps just that task."""
        kid = key_id(api_key)
        self.store.register_key(kid, key_hint(api_key), label)
        self.store.set_budget(kid, task, {
            "max_cost_usd": max_cost_usd, "max_total_tokens": max_total_tokens,
            "max_input_tokens": max_input_tokens, "max_cached_tokens": max_cached_tokens,
            "max_output_tokens": max_output_tokens}, period)

    def remove_budget(self, api_key: str, task: str = AGGREGATE) -> bool:
        return self.store.remove_budget(key_id(api_key), task)

    def reset_usage(self, api_key: str, task: str | None = None) -> int:
        return self.store.reset(key_id(api_key), task)

    def status(self, api_key: str | None = None, *, kid: str | None = None) -> list[dict]:
        """Used / limit / remaining for every budget (of one key, or all keys)."""
        kid = key_id(api_key) if api_key else kid
        rows = []
        for b in self.store.budgets(kid):
            used = self.store.usage(b["key_id"], b["task"], window_start(b["period"]))
            limits = {}
            for dim, col in LIMITS.items():
                if b[col] is not None:
                    lim = b[col]
                    limits[dim] = {"used": used[dim], "limit": lim, "remaining": max(lim - used[dim], 0),
                                   "pct_used": round(100 * used[dim] / lim, 2) if lim else 100.0,
                                   "exhausted": used[dim] >= lim}
            rows.append({"key": b["hint"], "key_id": b["key_id"], "label": b["label"], "task": b["task"],
                         "period": b["period"], "requests": used["requests"], "cost_usd": used["cost_usd"],
                         "input_tokens": used["input_tokens"], "cached_tokens": used["cached_tokens"],
                         "output_tokens": used["output_tokens"], "limits": limits,
                         "exhausted": any(x["exhausted"] for x in limits.values())})
        return rows

    def cost(self, model: str, input_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0) -> Cost:
        """Calculator: input_tokens are uncached input tokens."""
        return self.prices.cost(model, input_tokens, cached_tokens, output_tokens)

    # ---- request lifecycle ---------------------------------------------
    def _cost(self, model: str, u: Usage) -> Cost:
        try:
            return self.prices.cost(model, u.input_tokens, u.cached_tokens, u.output_tokens)
        except PricingError:
            if self.unknown_model == "block":
                raise
            log.warning("no price for model %r; recording tokens with $0 cost", model)
            return Cost(0.0, 0.0, 0.0)

    @staticmethod
    def _values(u: Usage, c: Cost) -> dict:
        return {"input_tokens": u.input_tokens, "cached_tokens": u.cached_tokens, "output_tokens": u.output_tokens,
                "total_tokens": u.total_tokens, "input_cost_usd": c.input_usd, "cached_cost_usd": c.cached_usd,
                "output_cost_usd": c.output_usd, "cost_usd": c.total_usd}

    def estimate(self, endpoint: str, body: dict) -> Usage:
        """Worst-case usage: all prompt tokens uncached, output at its max."""
        model = body.get("model", "")
        parts: list[str] = []
        n_ints = 0
        for field in ("messages", "input", "instructions", "prompt", "tools", "functions", "text", "response_format"):
            val = body.get(field)
            if endpoint == "embeddings" and isinstance(val, list) and val and not isinstance(val[0], str):
                # token-id input: [1, 2, 3] or [[1, 2], [3]]
                n_ints += sum(len(v) if isinstance(v, list) else 1 for v in val)
                continue
            _collect_text(val, parts)
        n_in = n_ints + (_count_tokens(model, "\n".join(parts)) if parts else 0)
        n_in += 4 * len(body.get("messages") or [])
        if endpoint == "embeddings":
            return Usage(n_in, 0, 0)
        out = next((body[k] for k in ("max_completion_tokens", "max_output_tokens", "max_tokens")
                    if body.get(k) is not None), self.default_output_reserve)
        return Usage(n_in, 0, int(out) * int(body.get("n") or 1))

    def reserve(self, api_key: str, task: str, endpoint: str, body: dict) -> Reservation:
        """Refuse (BudgetExceeded / PricingError) or reserve the worst-case estimate in the ledger."""
        kid, hint = key_id(api_key), key_hint(api_key)
        self.store.register_key(kid, hint)
        model = body.get("model", "")
        est = self.estimate(endpoint, body)
        row_id, violation = self.store.reserve(kid, task, model, endpoint, self._values(est, self._cost(model, est)),
                                               self.preflight, self.require_budget)
        if violation:
            raise BudgetExceeded(_message(violation, hint), violation, hint, task)
        return Reservation(row_id, kid, task, model, endpoint, est)

    @staticmethod
    def parse_usage(usage: dict) -> Usage:
        """Chat/Completions/Embeddings (`prompt_tokens`) and Responses (`input_tokens`) usage -> Usage."""
        if "input_tokens" in usage:
            total_in = usage.get("input_tokens") or 0
            cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
            out = usage.get("output_tokens") or 0
        else:
            total_in = usage.get("prompt_tokens") or 0
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            out = usage.get("completion_tokens") or 0
        return Usage(total_in - cached, cached, out)

    def finalize(self, res: Reservation, usage: dict | None) -> Usage | None:
        """Replace the reservation with actual usage; with no usage the estimate stays, marked 'estimated'."""
        if not usage:
            self.store.mark(res.row_id, "estimated")
            log.warning("no usage returned for %s request (task %s); kept worst-case estimate", res.endpoint, res.task)
            return None
        u = self.parse_usage(usage)
        self.store.finalize(res.row_id, self._values(u, self._cost(res.model, u)))
        return u

    def cancel(self, res: Reservation) -> None:
        """Upstream failed (not billed): drop the reservation."""
        self.store.cancel(res.row_id)
