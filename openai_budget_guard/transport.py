"""httpx transports that enforce budgets on OpenAI API calls.

Plug into any httpx-based client (the official openai SDK, LangChain's ChatOpenAI, ...). A refused request never
reaches OpenAI; the caller gets HTTP 429 with an OpenAI-style error body (code `cost_budget_exhausted`,
`token_budget_exhausted` or `no_budget_allotted`) and `x-should-retry: false`, so the SDK raises
`openai.RateLimitError` immediately instead of retrying.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from ._http import httpx
from .guard import DEFAULT_TASK, TASK_HEADER, BudgetExceeded, BudgetGuard, Reservation
from .pricing import PricingError

GUARDED_ENDPOINTS = ("chat/completions", "responses", "embeddings", "completions")
_STRIP_RESPONSE_HEADERS = ("content-encoding", "content-length", "transfer-encoding")


def _endpoint(request: httpx.Request) -> str | None:
    if request.method != "POST":
        return None
    path = request.url.path.rstrip("/")
    # longest first so ".../chat/completions" is not taken for "completions"
    return next((e for e in GUARDED_ENDPOINTS if path.endswith("/" + e)), None)


def _bytes_response(status: int, headers, raw: bytes, extensions: dict | None = None) -> httpx.Response:
    # unread ByteStream (not content=) so callers can still consume it with aiter_raw(), as the proxy does
    headers = httpx.Headers(headers)
    headers["content-length"] = str(len(raw))
    return httpx.Response(status, headers=headers, stream=httpx.ByteStream(raw), extensions=extensions or {})


def _error_response(status: int, body: dict) -> httpx.Response:
    return _bytes_response(status, {"content-type": "application/json", "x-should-retry": "false"},
                           json.dumps(body).encode())


@dataclass
class _Prepared:
    request: httpx.Request
    reservation: Reservation | None = None
    blocked: httpx.Response | None = None
    drop_usage_chunk: bool = False


def _prepare(request: httpx.Request, default_task: str, reserve) -> _Prepared:
    endpoint = _endpoint(request)
    task = request.headers.get(TASK_HEADER) or default_task
    headers = request.headers.copy()
    headers.pop(TASK_HEADER, None)
    if endpoint is None:
        if TASK_HEADER in request.headers:
            request = httpx.Request(request.method, request.url, headers=headers, content=request.content,
                                    extensions=request.extensions)
        return _Prepared(request)

    try:
        body = json.loads(request.content or b"{}")
    except (ValueError, httpx.RequestNotRead):
        return _Prepared(request)
    auth = request.headers.get("authorization", "")
    api_key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("api-key", "")
    if not api_key:
        return _Prepared(request)

    content = request.content
    drop = False
    if body.get("stream") and endpoint in ("chat/completions", "completions"):
        opts = dict(body.get("stream_options") or {})
        if not opts.get("include_usage"):
            # ask for the final usage chunk; strip it again so callers that did not ask never see it
            opts["include_usage"] = True
            body["stream_options"] = opts
            content, drop = json.dumps(body).encode(), True

    try:
        res = reserve(api_key, task, endpoint, body)
    except BudgetExceeded as e:
        return _Prepared(request, blocked=_error_response(429, e.to_openai_error()))
    except PricingError as e:
        return _Prepared(request, blocked=_error_response(400, {"error": {
            "message": str(e), "type": "invalid_request_error", "code": "unknown_model_pricing", "param": "model"}}))

    headers.pop("content-length", None)
    headers["accept-encoding"] = "identity"   # keep SSE / JSON bodies uncompressed so usage can be read
    request = httpx.Request(request.method, request.url, headers=headers, content=content,
                            extensions=request.extensions)
    return _Prepared(request, res, drop_usage_chunk=drop)


class _SSEObserver:
    """Splits an SSE byte stream into events, captures the final `usage`, optionally drops the usage-only chunk."""

    def __init__(self, drop_usage_chunk: bool):
        self.drop = drop_usage_chunk
        self.buf = b""
        self.usage: dict | None = None

    def feed(self, chunk: bytes) -> bytes:
        self.buf += chunk
        out = []
        while True:
            cuts = [(i, len(s)) for s in (b"\n\n", b"\r\n\r\n") if (i := self.buf.find(s)) != -1]
            if not cuts:
                break
            i, n = min(cuts)
            event, self.buf = self.buf[:i + n], self.buf[i + n:]
            if self._keep(event):
                out.append(event)
        return b"".join(out)

    def flush(self) -> bytes:
        rest, self.buf = self.buf, b""
        return rest if rest and self._keep(rest) else b""

    def _keep(self, event: bytes) -> bool:
        for line in event.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("usage"):                                        # chat / completions final chunk
                self.usage = obj["usage"]
                if self.drop and obj.get("choices") == []:
                    return False
            resp = obj.get("response")
            if isinstance(resp, dict) and resp.get("usage"):            # responses: response.completed etc.
                self.usage = resp["usage"]
        return True


def _passthrough_headers(response: httpx.Response) -> httpx.Headers:
    headers = response.headers.copy()
    for h in _STRIP_RESPONSE_HEADERS:
        headers.pop(h, None)
    return headers


class _SyncSSEStream(httpx.SyncByteStream):
    def __init__(self, response: httpx.Response, observer: _SSEObserver, done):
        self.response, self.observer, self.done = response, observer, done
        self.finished = False

    def __iter__(self):
        for chunk in self.response.iter_bytes():
            out = self.observer.feed(chunk)
            if out:
                yield out
        tail = self.observer.flush()
        if tail:
            yield tail

    def close(self):
        try:
            self.response.close()
        finally:
            if not self.finished:
                self.finished = True
                self.done(self.observer.usage)


class _AsyncSSEStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response, observer: _SSEObserver, done):
        self.response, self.observer, self.done = response, observer, done
        self.finished = False

    async def __aiter__(self):
        async for chunk in self.response.aiter_bytes():
            out = self.observer.feed(chunk)
            if out:
                yield out
        tail = self.observer.flush()
        if tail:
            yield tail

    async def aclose(self):
        try:
            await self.response.aclose()
        finally:
            if not self.finished:
                self.finished = True
                await asyncio.to_thread(self.done, self.observer.usage)


def _default_inner_kwargs() -> dict:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return {"proxy": proxy} if proxy else {}


class BudgetTransport(httpx.BaseTransport):
    def __init__(self, guard: BudgetGuard, default_task: str = DEFAULT_TASK,
                 inner: httpx.BaseTransport | None = None):
        self.guard = guard
        self.default_task = default_task
        self.inner = inner or httpx.HTTPTransport(**_default_inner_kwargs())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        prep = _prepare(request, self.default_task, self.guard.reserve)
        if prep.blocked is not None:
            return prep.blocked
        res = prep.reservation
        try:
            response = self.inner.handle_request(prep.request)
        except BaseException:
            if res:
                self.guard.cancel(res)
            raise
        if res is None:
            return response
        if response.status_code >= 400:
            self.guard.cancel(res)
            return response
        if "text/event-stream" in response.headers.get("content-type", ""):
            stream = _SyncSSEStream(response, _SSEObserver(prep.drop_usage_chunk),
                                    lambda usage: self.guard.finalize(res, usage))
            return httpx.Response(response.status_code, headers=_passthrough_headers(response), stream=stream,
                                  extensions=response.extensions)
        try:
            raw = response.read()
        finally:
            response.close()
        try:
            usage = json.loads(raw).get("usage")
        except (ValueError, AttributeError):
            usage = None
        self.guard.finalize(res, usage)
        return _bytes_response(response.status_code, _passthrough_headers(response), raw, response.extensions)

    def close(self) -> None:
        self.inner.close()


class AsyncBudgetTransport(httpx.AsyncBaseTransport):
    def __init__(self, guard: BudgetGuard, default_task: str = DEFAULT_TASK,
                 inner: httpx.AsyncBaseTransport | None = None):
        self.guard = guard
        self.default_task = default_task
        self.inner = inner or httpx.AsyncHTTPTransport(**_default_inner_kwargs())

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        prep = await asyncio.to_thread(_prepare, request, self.default_task, self.guard.reserve)
        if prep.blocked is not None:
            return prep.blocked
        res = prep.reservation
        try:
            response = await self.inner.handle_async_request(prep.request)
        except BaseException:
            if res:
                await asyncio.to_thread(self.guard.cancel, res)
            raise
        if res is None:
            return response
        if response.status_code >= 400:
            await asyncio.to_thread(self.guard.cancel, res)
            return response
        if "text/event-stream" in response.headers.get("content-type", ""):
            stream = _AsyncSSEStream(response, _SSEObserver(prep.drop_usage_chunk),
                                     lambda usage: self.guard.finalize(res, usage))
            return httpx.Response(response.status_code, headers=_passthrough_headers(response), stream=stream,
                                  extensions=response.extensions)
        try:
            raw = await response.aread()
        finally:
            await response.aclose()
        try:
            usage = json.loads(raw).get("usage")
        except (ValueError, AttributeError):
            usage = None
        await asyncio.to_thread(self.guard.finalize, res, usage)
        return _bytes_response(response.status_code, _passthrough_headers(response), raw, response.extensions)

    async def aclose(self) -> None:
        await self.inner.aclose()
