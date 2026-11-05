"""Pick the httpx module the installed openai SDK is built on.

openai<3 uses `httpx`; openai>=3 vendors the same API as `httpx2`. A transport must subclass the module the SDK's
client actually uses, so everything in this package (and its tests) imports httpx from here.
"""
from __future__ import annotations

import importlib


def _sdk_httpx():
    try:
        import openai._base_client as base
    except ImportError:  # pragma: no cover - openai missing
        return importlib.import_module("httpx")
    for name in ("httpx2", "httpx"):
        mod = getattr(base, name, None)
        if mod is not None:
            return mod
    return importlib.import_module("httpx")  # pragma: no cover


httpx = _sdk_httpx()

__all__ = ["httpx"]
