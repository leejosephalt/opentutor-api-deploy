"""Managed, admin-controlled LLM provider.

When ``MANAGED_LLM_ENABLED`` is on, OpenTutor ignores user provider settings and
serves AI through this layer only:

* **Fast**  -> the small/free model (default ``particle-1-free``)
* **Smart** -> the larger free model (default ``xiaomi-mimo-v2.6-pro-free``)

Each tier tries the primary provider (AiHubMix) and, if that provider is rate
limited / exhausted / unhealthy, transparently falls back to OpenRouter's free
model. If both are unavailable the call raises :class:`AIExhaustedError`, which
the API surfaces as a clear "AI is fully used up on Free mode" message.

The provider API keys never leave the server.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from libs.exceptions import AppError
from services.llm.base_client import LLMClient

logger = logging.getLogger(__name__)

AI_EXHAUSTED_MESSAGE = (
    "The AI is fully used up on Free mode right now. Please try again later."
)


class AIExhaustedError(AppError):
    """Both the primary and backup providers are unavailable/exhausted."""

    code = "ai_exhausted"
    status = 429

    def __init__(self, message: str = AI_EXHAUSTED_MESSAGE):
        super().__init__(message)


class ManagedLLMClient(LLMClient):
    """A tier-bound client that fails over from primary to backup."""

    provider_name = "managed"

    def __init__(self, *, tier: str, primary: LLMClient, fallback: LLMClient | None = None):
        super().__init__()
        self.tier = tier
        self._primary = primary
        self._fallback = fallback
        self.provider_name = f"managed-{tier}"

    def _candidates(self) -> list[LLMClient]:
        candidates: list[LLMClient] = []
        if self._primary is not None and self._primary.is_healthy:
            candidates.append(self._primary)
        if self._fallback is not None and self._fallback.is_healthy:
            candidates.append(self._fallback)
        return candidates

    def _record_usage(self, result: object) -> None:
        if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict):
            self._last_usage = result[1]

    async def chat(self, system_prompt: str, user_message: str, images: list[dict] | None = None):
        last_error: Exception | None = None
        for client in self._candidates():
            try:
                result = await client.chat(system_prompt, user_message, images)
                self._record_usage(result)
                return result
            except Exception as exc:  # noqa: BLE001 - fail over across providers
                last_error = exc
                logger.warning(
                    "Managed %s tier: %s failed (%s); trying backup",
                    self.tier,
                    client.provider_name,
                    type(exc).__name__,
                )
        raise AIExhaustedError() from last_error

    async def extract(self, system_prompt: str, user_message: str):
        last_error: Exception | None = None
        for client in self._candidates():
            try:
                result = await client.extract(system_prompt, user_message)
                self._record_usage(result)
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Managed %s tier: %s failed (%s); trying backup",
                    self.tier,
                    client.provider_name,
                    type(exc).__name__,
                )
        raise AIExhaustedError() from last_error

    async def stream_chat(
        self,
        system_prompt: str,
        user_message: str,
        images: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        last_error: Exception | None = None
        for client in self._candidates():
            yielded = False
            try:
                async for chunk in client.stream_chat(system_prompt, user_message, images):
                    yielded = True
                    yield chunk
                self._last_usage = client.get_last_usage()
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if yielded:
                    # The stream already reached the client; can't safely restart it.
                    raise
                logger.warning(
                    "Managed %s tier: %s stream failed (%s); trying backup",
                    self.tier,
                    client.provider_name,
                    type(exc).__name__,
                )
        raise AIExhaustedError() from last_error
