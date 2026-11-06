"""End-to-end tests through the real openai SDK against a mocked OpenAI server (no network, no spend)."""
import asyncio
import json

import openai
import pytest

from openai_budget_guard import BudgetGuard, for_task, guarded_async_openai, guarded_openai
from openai_budget_guard._http import httpx

KEY = "sk-test-0000000000000000abcd"
OTHER_KEY = "sk-test-1111111111111111wxyz"


class FakeOpenAI:
    """Mock upstream. Chat usage: 1000 prompt (400 cached) + 200 completion tokens."""

    def __init__(self):
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((request.url.path, body, dict(request.headers)))
        path = request.url.path
        if "FAIL" in json.dumps(body):
            return httpx.Response(500, json={"error": {"message": "boom"}})
        if path.endswith("/chat/completions"):
            usage = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                     "prompt_tokens_details": {"cached_tokens": 400}}
            if body.get("stream"):
                chunks = [{"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                           "choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}],
                           **({"usage": None} if body.get("stream_options", {}).get("include_usage") else {})}
                          for w in ("Hel", "lo")]
                if body.get("stream_options", {}).get("include_usage"):
                    chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                   "model": body["model"], "choices": [], "usage": usage})
                sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={
                "id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"},
                             "finish_reason": "stop"}], "usage": usage})
        if path.endswith("/responses"):
            return httpx.Response(200, json={
                "id": "r", "object": "response", "created_at": 0, "model": body["model"], "status": "completed",
                "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 500, "input_tokens_details": {"cached_tokens": 100},
                          "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 0},
                          "total_tokens": 550}})
        if path.endswith("/embeddings"):
            return httpx.Response(200, json={"object": "list", "model": body["model"],
                                             "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                                             "usage": {"prompt_tokens": 10, "total_tokens": 10}})
        return httpx.Response(200, json={"object": "list", "data": []})


@pytest.fixture
def env(tmp_path):
    fake = FakeOpenAI()
    guard = BudgetGuard(tmp_path / "usage.db")
    return guard, fake


def client(guard, fake, task="default", key=KEY):
    return guarded_openai(key, task=task, guard=guard, inner=httpx.MockTransport(fake))


def chat(c, **kw):
    return c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], **kw)


def test_records_input_cached_output_and_cost(env):
    guard, fake = env
    r = chat(client(guard, fake))
    assert r.choices[0].message.content == "Hello"
    [row] = guard.store.recent()
    assert (row["input_tokens"], row["cached_tokens"], row["output_tokens"]) == (600, 400, 200)
    # gpt-4o: 2.50 in / 1.25 cached / 10.00 out per 1M
    assert row["cost_usd"] == pytest.approx(600 * 2.5e-6 + 400 * 1.25e-6 + 200 * 10e-6)
    assert row["status"] == "final"


def test_cost_budget_exhausted_blocks_without_retry(env):
    guard, fake = env
    guard.set_budget(KEY, max_cost_usd=0.0035)           # one call costs $0.004
    c = client(guard, fake)
    chat(c, max_tokens=200)
    with pytest.raises(openai.RateLimitError) as e:
        chat(c, max_tokens=200)
    assert e.value.code == "cost_budget_exhausted"
    assert "Cost budget exhausted" in str(e.value) and "all tasks combined" in str(e.value)
    assert len(fake.calls) == 1                           # blocked call never reached OpenAI, and no SDK retries


def test_task_budget_is_individual_aggregate_is_shared(env):
    guard, fake = env
    guard.set_budget(KEY, "summarize", max_output_tokens=300)
    guard.set_budget(KEY, "*", max_total_tokens=2450)
    c = client(guard, fake, task="summarize")
    chat(c, max_tokens=100)
    with pytest.raises(openai.RateLimitError) as e:      # 200 used + 200 requested > 300
        chat(c, max_tokens=200)
    assert e.value.code == "token_budget_exhausted" and "task 'summarize'" in str(e.value)
    qa = for_task(c, "qa")                                # other task unaffected by summarize's cap...
    chat(qa, max_tokens=100)
    with pytest.raises(openai.RateLimitError) as e:      # ...until the shared aggregate (2400 used + ~105 est > 2450) runs out
        chat(qa, max_tokens=100)
    assert "all tasks combined" in str(e.value)
    assert {r["task"] for r in guard.store.recent()} == {"summarize", "qa"}
    assert all("x-budget-task" not in {k.lower() for k in h} for _, _, h in fake.calls)


def test_keys_are_isolated(env):
    guard, fake = env
    guard.set_budget(KEY, max_cost_usd=0)
    with pytest.raises(openai.RateLimitError):
        chat(client(guard, fake))
    chat(client(guard, fake, key=OTHER_KEY))


def test_preflight_refuses_request_that_would_overshoot(env):
    guard, fake = env
    guard.set_budget(KEY, max_output_tokens=1000)
    with pytest.raises(openai.RateLimitError) as e:
        chat(client(guard, fake), max_tokens=5000)
    assert "insufficient" in str(e.value) and not fake.calls
    assert guard.store.recent() == []


def test_streaming_usage_recorded_and_injected_chunk_hidden(env):
    guard, fake = env
    chunks = list(chat(client(guard, fake), stream=True))
    assert [ch.choices[0].delta.content for ch in chunks] == ["Hel", "lo"]   # no empty-choices usage chunk
    assert fake.calls[0][1]["stream_options"] == {"include_usage": True}
    [row] = guard.store.recent()
    assert (row["input_tokens"], row["cached_tokens"], row["output_tokens"], row["status"]) == (600, 400, 200, "final")


def test_streaming_keeps_usage_chunk_when_caller_asked(env):
    guard, fake = env
    chunks = list(chat(client(guard, fake), stream=True, stream_options={"include_usage": True}))
    assert chunks[-1].usage.completion_tokens == 200


def test_responses_api_and_embeddings(env):
    guard, fake = env
    c = client(guard, fake)
    c.responses.create(model="gpt-5-mini", input="hello")
    c.embeddings.create(model="text-embedding-3-small", input=["a", "b"])
    emb, resp = guard.store.recent()
    assert (resp["input_tokens"], resp["cached_tokens"], resp["output_tokens"]) == (400, 100, 50)
    assert resp["cost_usd"] == pytest.approx(400 * 0.25e-6 + 100 * 0.025e-6 + 50 * 2e-6)
    assert (emb["input_tokens"], emb["output_tokens"]) == (10, 0)


def test_upstream_error_releases_reservation(env):
    guard, fake = env
    c = guarded_openai(KEY, guard=guard, inner=httpx.MockTransport(fake), max_retries=0)
    with pytest.raises(openai.InternalServerError):
        c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "FAIL"}])
    assert guard.store.recent() == []


def test_unknown_model_is_refused(env):
    guard, fake = env
    with pytest.raises(openai.BadRequestError) as e:
        client(guard, fake).chat.completions.create(model="mystery-1", messages=[{"role": "user", "content": "x"}])
    assert e.value.code == "unknown_model_pricing" and not fake.calls


def test_async_client(env):
    guard, fake = env
    guard.set_budget(KEY, max_cost_usd=0.0035)

    async def run():
        c = guarded_async_openai(KEY, guard=guard, inner=httpx.MockTransport(fake))
        await c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=200)
        stream = await c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
                                                 stream=True, max_tokens=1)
        return [ch async for ch in stream]

    with pytest.raises(openai.RateLimitError):
        asyncio.run(run())
    assert len(fake.calls) == 1


def test_status_and_calculator(env):
    guard, fake = env
    guard.set_budget(KEY, max_cost_usd=1, period="daily", label="demo")
    chat(client(guard, fake))
    [s] = guard.status(KEY)
    assert s["label"] == "demo" and s["limits"]["cost_usd"]["used"] == pytest.approx(0.004)
    assert guard.cost("gpt-4o-2024-08-06", 1_000_000, 0, 0).total_usd == pytest.approx(2.5)


def test_proxy(env):
    from fastapi.testclient import TestClient

    from openai_budget_guard.proxy import create_app

    guard, fake = env
    guard.set_budget(KEY, max_cost_usd=0.0035)
    app = create_app(guard, upstream="https://api.openai.com/v1", inner=httpx.MockTransport(fake))
    with TestClient(app) as tc:
        oc = openai.OpenAI(api_key=KEY, base_url="http://testserver/v1", http_client=tc)
        assert [ch.choices[0].delta.content for ch in chat(oc, stream=True, max_tokens=200)] == ["Hel", "lo"]
        with pytest.raises(openai.RateLimitError) as e:
            chat(oc, max_tokens=200)
        assert e.value.code == "cost_budget_exhausted"
        # The proxy settles a stream's actual usage in a background task that runs after the last chunk is
        # sent; until then the request's (smaller) reservation is what the ledger holds. Wait for settlement.
        import time

        for _ in range(40):
            st = tc.get("/budget/status", headers={"Authorization": f"Bearer {KEY}"}).json()
            if st[0]["exhausted"]:
                break
            time.sleep(0.05)
        assert st[0]["exhausted"] is True


def test_concurrent_requests_cannot_overshoot(env):
    from concurrent.futures import ThreadPoolExecutor

    guard, fake = env
    guard.set_budget(KEY, max_output_tokens=1000)        # each call reserves and uses 200 output tokens
    c = client(guard, fake)

    def call(_):
        try:
            chat(c, max_tokens=200)
            return True
        except openai.RateLimitError:
            return False

    with ThreadPoolExecutor(20) as pool:
        ok = list(pool.map(call, range(20)))
    assert sum(ok) == 5
    assert guard.store.usage(guard.store.keys()[0]["key_id"])["output_tokens"] == 1000
