"""
LLM client abstraction with OpenAI and Anthropic backends.

Both clients expose the same `complete_json(system, user, seed)` interface,
so the rest of the pipeline is provider-agnostic.  Use `make_client()` to
instantiate the right backend based on the model name.

  make_client("gpt-4o", ...)          → LLMClient (OpenAI)
  make_client("claude-opus-4-7", ...) → AnthropicLLMClient

Shared features (both backends):
  - Exponential-backoff retry on transient errors
  - Token-bucket rate limiting
  - SHA-256 prompt cache (disk-backed)
  - JSON extraction and repair from raw LLM text
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from src.utils.hashing import sha256_text
from src.utils.io import DiskCache

log = logging.getLogger("pipeline.llm")


# ── Shared base ───────────────────────────────────────────────────────────────

class _BaseLLMClient:
    """Cache, rate limiting, retry loop, and JSON repair — provider-agnostic."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        requests_per_minute: int = 30,
        cache_dir: Path | None = None,
        enable_cache: bool = True,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._min_interval = 60.0 / requests_per_minute
        self._last_call: float = 0.0
        self._sem = asyncio.Semaphore(5)

        self._cache: DiskCache | None = None
        if enable_cache and cache_dir:
            self._cache = DiskCache(cache_dir)

    async def complete_json(
        self,
        system: str,
        user: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        cache_key = sha256_text(f"{self.model}|{self.temperature}|{system}|{user}")

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                log.debug("Cache hit: %s", cache_key[:12])
                return cached

        raw = await self._call_with_retry(system, user, seed)
        parsed = self._extract_json(raw)

        if self._cache:
            self._cache.set(cache_key, parsed)

        return parsed

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats if self._cache else {}

    async def _call_with_retry(self, system: str, user: str, seed: int | None) -> str:
        raise NotImplementedError

    async def _rate_limited_call(self, coro_fn, *args, **kwargs) -> str:
        """Enforce minimum inter-request interval then call coro_fn."""
        async with self._sem:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()
            return await coro_fn(*args, **kwargs)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in (r'\{.*\}', r'\[.*\]'):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        repaired = text.rstrip()
        for closing in ("}", "]", '"}', '"]}'):
            try:
                return json.loads(repaired + closing)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM output:\n{text[:500]}")


# ── OpenAI backend ────────────────────────────────────────────────────────────

class LLMClient(_BaseLLMClient):
    """
    OpenAI-compatible backend (OpenAI, Together AI, Mistral, Ollama, …).
    Set OPENAI_BASE_URL in .env to redirect to an alternate provider.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()   # reads OPENAI_API_KEY / OPENAI_BASE_URL

    async def _call_with_retry(self, system: str, user: str, seed: int | None) -> str:
        import openai
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self._rate_limited_call(self._call, system, user, seed)
            except (openai.RateLimitError, openai.APIStatusError) as exc:
                wait = self.retry_delay * (2 ** attempt)
                log.warning("OpenAI error (attempt %d/%d): %s — retry in %.1fs",
                            attempt + 1, self.max_retries, exc, wait)
                await asyncio.sleep(wait)
                last_exc = exc
            except openai.APIConnectionError as exc:
                log.warning("OpenAI connection error (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, exc)
                await asyncio.sleep(self.retry_delay)
                last_exc = exc
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} attempts") from last_exc

    async def _call(self, system: str, user: str, seed: int | None) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        if seed is not None:
            kwargs["seed"] = seed
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ── Anthropic backend ─────────────────────────────────────────────────────────

class AnthropicLLMClient(_BaseLLMClient):
    """
    Anthropic backend for Claude models (claude-opus-*, claude-sonnet-*, …).
    Reads ANTHROPIC_API_KEY from the environment.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for Claude models. "
                "Run: pip install anthropic"
            ) from exc
        self._client = AsyncAnthropic()   # reads ANTHROPIC_API_KEY

    async def _call_with_retry(self, system: str, user: str, seed: int | None) -> str:
        import anthropic
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self._rate_limited_call(self._call, system, user)
            except anthropic.RateLimitError as exc:
                wait = self.retry_delay * (2 ** attempt)
                log.warning("Anthropic rate limit (attempt %d/%d) — retry in %.1fs",
                            attempt + 1, self.max_retries, wait)
                await asyncio.sleep(wait)
                last_exc = exc
            except anthropic.APIStatusError as exc:
                wait = self.retry_delay * (2 ** attempt)
                log.warning("Anthropic API error (attempt %d/%d): %s — retry in %.1fs",
                            attempt + 1, self.max_retries, exc, wait)
                await asyncio.sleep(wait)
                last_exc = exc
            except anthropic.APIConnectionError as exc:
                log.warning("Anthropic connection error (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, exc)
                await asyncio.sleep(self.retry_delay)
                last_exc = exc
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} attempts") from last_exc

    async def _call(self, system: str, user: str) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# ── Factory ───────────────────────────────────────────────────────────────────

def make_client(model: str, **kwargs) -> _BaseLLMClient:
    """
    Return the appropriate client for the given model name.

    Claude models (prefix "claude-") → AnthropicLLMClient
    Everything else                  → LLMClient (OpenAI-compatible)
    """
    if model.startswith("claude"):
        log.info("Using Anthropic backend for model: %s", model)
        return AnthropicLLMClient(model=model, **kwargs)
    log.info("Using OpenAI backend for model: %s", model)
    return LLMClient(model=model, **kwargs)
