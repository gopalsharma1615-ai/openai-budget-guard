# Changelog

## 0.2.1 — 2026-11-06
- Fix: a streamed request is settled as soon as its final usage chunk is observed, not only when the stream
  is closed. A consumer that stops reading at `[DONE]` or disconnects early (which can skip the proxy's
  background close) no longer leaves the request recorded at its smaller estimated cost.

## 0.2.0 — 2026-11-05
- Works with openai 1.x, 2.x and 3.x: transports now subclass whichever httpx module the installed SDK uses
  (`httpx` for openai<3, `httpx2` for openai>=3), via `openai_budget_guard._http`.
- Price table refreshed against the OpenAI pricing page (2026-11-05) for GPT-6, GPT-5.6/5.5/5.4/5.2 families;
  `_verified` date added. Long-context, Batch and Flex rates are not modelled.
- Packaging: `pyproject.toml`, `obg` console script, `[proxy]` extra, MIT licence, CI matrix, GitHub Pages docs.

## 0.1.0
- Per-key / per-task budgets (cost, total, input, cached, output tokens) with hourly→monthly windows.
- Atomic preflight reservations, streaming usage capture, SQLite WAL ledger, proxy, CLI.
