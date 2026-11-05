"""Model price book: USD per 1M tokens for uncached input, cached input and output."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRICING_PATH = Path(__file__).with_name("pricing.json")


class PricingError(LookupError):
    """Raised when a model has no configured price."""


@dataclass(frozen=True)
class Price:
    input: float          # USD / 1M uncached input tokens
    cached_input: float   # USD / 1M cached input tokens
    output: float         # USD / 1M output tokens


@dataclass(frozen=True)
class Cost:
    input_usd: float
    cached_usd: float
    output_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.cached_usd + self.output_usd

    def as_dict(self) -> dict:
        return {"input_usd": self.input_usd, "cached_usd": self.cached_usd,
                "output_usd": self.output_usd, "total_usd": self.total_usd}


def normalize_model(model: str) -> str:
    model = (model or "").strip()
    if model.startswith("ft:"):            # ft:gpt-4o-mini-2024-07-18:org::id
        model = model.split(":")[1]
    if "/" in model:                       # openai/gpt-4o
        model = model.rsplit("/", 1)[1]
    return model


class PriceBook:
    def __init__(self, prices: dict[str, Price], source: str = "<memory>"):
        self.prices = prices
        self.source = source

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> PriceBook:
        path = Path(path or os.environ.get("OBG_PRICING") or DEFAULT_PRICING_PATH)
        raw = json.loads(path.read_text())
        prices = {}
        for name, p in raw.get("models", raw).items():
            if name.startswith("_"):
                continue
            inp = float(p["input"])
            cached = p.get("cached_input")
            prices[name] = Price(inp, inp if cached is None else float(cached), float(p.get("output", 0.0)))
        return cls(prices, str(path))

    def lookup(self, model: str) -> Price:
        name = normalize_model(model)
        if name in self.prices:
            return self.prices[name]
        best = max((m for m in self.prices if name.startswith(m)), key=len, default=None)
        if best is None:
            raise PricingError(f"No pricing configured for model '{model}'. Add it to {self.source} "
                               f"or point OBG_PRICING at a price file that has it.")
        return self.prices[best]

    def cost(self, model: str, input_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0) -> Cost:
        """input_tokens are *uncached* input tokens; cached_tokens are billed at the cached rate."""
        p = self.lookup(model)
        return Cost(input_tokens * p.input / 1e6, cached_tokens * p.cached_input / 1e6, output_tokens * p.output / 1e6)
