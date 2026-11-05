"""OpenAI-compatible budget proxy for apps in any language.

Point the app at  base_url = http://<host>:<port>/v1  and keep using its own OpenAI key. The proxy checks the
key's budgets, forwards to OpenAI, meters usage and returns OpenAI's response unchanged (or a 429 budget error).
Pick the task per request with the `X-Budget-Task` header.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from ._http import httpx
from .guard import DEFAULT_TASK, BudgetGuard
from .transport import AsyncBudgetTransport

_HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade",
                "proxy-authorization", "te", "trailer"}


def create_app(guard: BudgetGuard | None = None, upstream: str = "https://api.openai.com/v1",
               default_task: str = DEFAULT_TASK, inner: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    guard = guard or BudgetGuard()
    client = httpx.AsyncClient(transport=AsyncBudgetTransport(guard, default_task, inner=inner),
                               timeout=httpx.Timeout(600, connect=10))

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await client.aclose()

    app = FastAPI(title="OpenAI Budget Guard", lifespan=lifespan)

    @app.get("/budget/status")
    def budget_status(authorization: str = Header(...)):
        """Budgets and usage for the key in the Authorization header (a key can only see itself)."""
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Authorization: Bearer <openai key> required")
        return guard.status(authorization[7:].strip())

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def forward(path: str, request: Request):
        url = f"{upstream.rstrip('/')}/{path}"
        if request.url.query:
            url += "?" + request.url.query
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        req = client.build_request(request.method, url, headers=headers, content=await request.body())
        resp = await client.send(req, stream=True)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_HEADERS}
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=out_headers,
                                 background=BackgroundTask(resp.aclose))

    return app


def serve(host: str = "127.0.0.1", port: int = 8787, **kwargs) -> None:
    import uvicorn

    uvicorn.run(create_app(**kwargs), host=host, port=port)
