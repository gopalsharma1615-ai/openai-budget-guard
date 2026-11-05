"""Drop-in OpenAI clients with budget enforcement."""
from __future__ import annotations

import os

from .guard import DEFAULT_TASK, TASK_HEADER, BudgetGuard
from .transport import AsyncBudgetTransport, BudgetTransport


def guarded_openai(api_key: str | None = None, *, task: str = DEFAULT_TASK, guard: BudgetGuard | None = None,
                   inner=None, **client_kwargs):
    """openai.OpenAI whose every request is checked against the key's budgets and metered."""
    import openai

    api_key = api_key or os.environ["OPENAI_API_KEY"]
    http_client = openai.DefaultHttpxClient(transport=BudgetTransport(guard or BudgetGuard(), task, inner=inner))
    return openai.OpenAI(api_key=api_key, http_client=http_client, **client_kwargs)


def guarded_async_openai(api_key: str | None = None, *, task: str = DEFAULT_TASK,
                         guard: BudgetGuard | None = None, inner=None, **client_kwargs):
    """openai.AsyncOpenAI counterpart of guarded_openai."""
    import openai

    api_key = api_key or os.environ["OPENAI_API_KEY"]
    http_client = openai.DefaultAsyncHttpxClient(
        transport=AsyncBudgetTransport(guard or BudgetGuard(), task, inner=inner))
    return openai.AsyncOpenAI(api_key=api_key, http_client=http_client, **client_kwargs)


def for_task(client, task: str):
    """Same client (shared connection pool) billed to another task."""
    return client.with_options(default_headers={TASK_HEADER: task})
