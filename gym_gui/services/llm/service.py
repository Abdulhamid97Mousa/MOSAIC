"""LLM service supporting OpenRouter (cloud) and vLLM (local).

Both providers use OpenAI-compatible API format:
- OpenRouter: https://openrouter.ai/docs/quickstart
- vLLM: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from gym_gui.services.llm.config import LLMConfig
from gym_gui.services.llm.models import LLMProvider, ModelIdentity

_LOGGER = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """LLM service errors."""

    pass


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class CompletionResult:
    """Result from a chat completion request."""

    content: str
    model_used: ModelIdentity
    tokens_used: Optional[int] = None
    cancelled: bool = False
    error: Optional[str] = None


class LLMService:
    """LLM service supporting OpenRouter and vLLM.

    Both providers use OpenAI-compatible chat completions API:
    - POST /v1/chat/completions
    - Authorization: Bearer <api_key>

    OpenRouter (cloud):
        - Base URL: https://openrouter.ai/api/v1
        - API Key: sk-or-v1-... (from openrouter.ai/keys)
        - Extra headers: HTTP-Referer, X-Title

    vLLM (local):
        - Base URL: http://localhost:8000/v1
        - API Key: Any string (or "EMPTY" for no auth)
        - Start server: vllm serve <model> --api-key token-abc123

    Example:
        >>> config = LLMConfig()
        >>> config.set_api_key("sk-or-v1-...")
        >>> service = LLMService(config)
        >>> result = await service.chat_completion(
        ...     messages=[ChatMessage(role="user", content="Hello!")],
        ...     model=config.preferred_models[0],
        ... )
        >>> print(result.content)
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def _get_api_key_for_model(self, model: ModelIdentity) -> Optional[str]:
        """Get the API key for a model's provider."""
        if model.provider == LLMProvider.OPENROUTER:
            if self.config.openrouter_api_key:
                return self.config.openrouter_api_key
            return os.getenv("OPENROUTER_API_KEY")
        elif model.provider == LLMProvider.VLLM:
            return self.config.vllm_api_key or "EMPTY"
        elif model.provider == LLMProvider.ZHIPU:
            if self.config.zhipu_api_key:
                return self.config.zhipu_api_key
            return os.getenv("ZHIPU_API_KEY") or os.getenv("ZAI_API_KEY")
        return None

    def _get_base_url_for_model(self, model: ModelIdentity) -> str:
        """Get the API base URL for a model's provider."""
        if model.provider == LLMProvider.OPENROUTER:
            return self.config.openrouter_base_url
        elif model.provider == LLMProvider.VLLM:
            return self.config.vllm_base_url
        elif model.provider == LLMProvider.ZHIPU:
            return self.config.zhipu_base_url
        return self.config.openrouter_base_url

    def _get_headers_for_model(self, model: ModelIdentity, api_key: str) -> Dict[str, str]:
        """Get HTTP headers for a model's provider."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter-specific headers for rankings
        if model.provider == LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = "https://github.com/mosaic-rl/mosaic"
            headers["X-Title"] = "MOSAIC RL Framework"
        return headers

    def _do_completion_sync(
        self,
        messages: List[ChatMessage],
        model: ModelIdentity,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Synchronous completion request (runs in thread pool)."""
        api_key = self._get_api_key_for_model(model)

        # Check API key for OpenRouter
        if model.provider == LLMProvider.OPENROUTER and not api_key:
            return CompletionResult(
                content="",
                model_used=model,
                error="No API key configured. Please enter your OpenRouter API key.",
            )

        # vLLM can use "EMPTY" as default
        if not api_key:
            api_key = "EMPTY"

        # Build URL: <base_url>/chat/completions
        base_url = self._get_base_url_for_model(model)
        url = f"{base_url}/chat/completions"

        headers = self._get_headers_for_model(model, api_key)

        data = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(data),
                timeout=self.config.request_timeout_seconds,
            )

            if response.status_code == 401:
                if model.provider == LLMProvider.OPENROUTER:
                    return CompletionResult(
                        content="",
                        model_used=model,
                        error="Invalid API key. Please check your OpenRouter API key.",
                    )
                else:
                    return CompletionResult(
                        content="",
                        model_used=model,
                        error="vLLM authentication failed. Check --api-key on server.",
                    )

            if response.status_code == 429:
                return CompletionResult(
                    content="",
                    model_used=model,
                    error="Rate limited. Please wait a moment and try again.",
                )

            if response.status_code != 200:
                error_text = response.text[:200] if response.text else "Unknown error"
                return CompletionResult(
                    content="",
                    model_used=model,
                    error=f"API error ({response.status_code}): {error_text}",
                )

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens")

            return CompletionResult(
                content=content,
                model_used=model,
                tokens_used=tokens,
            )

        except requests.exceptions.Timeout:
            return CompletionResult(
                content="",
                model_used=model,
                error="Request timed out. The model may be slow or unavailable.",
            )
        except requests.exceptions.ConnectionError:
            if model.provider == LLMProvider.VLLM:
                return CompletionResult(
                    content="",
                    model_used=model,
                    error=f"Cannot connect to vLLM at {base_url}. Is the server running?",
                )
            return CompletionResult(
                content="",
                model_used=model,
                error="Cannot connect to OpenRouter. Check your internet connection.",
            )
        except Exception as e:
            _LOGGER.exception("Chat completion failed")
            return CompletionResult(
                content="",
                model_used=model,
                error=f"Unexpected error: {str(e)}",
            )

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: ModelIdentity,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CompletionResult:
        """Send chat completion request to OpenRouter.

        Args:
            messages: List of chat messages.
            model: Model to use for completion.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.

        Returns:
            CompletionResult with response content.
        """
        try:
            # Run synchronous request in thread pool to not block Qt
            result = await asyncio.to_thread(
                self._do_completion_sync,
                messages,
                model,
                temperature or self.config.default_temperature,
                max_tokens or self.config.default_max_tokens,
            )
            return result
        except asyncio.CancelledError:
            return CompletionResult(
                content="",
                model_used=model,
                cancelled=True,
            )

    async def validate_api_key(self, api_key: str) -> tuple[bool, str]:
        """Validate an OpenRouter API key.

        Args:
            api_key: The API key to validate.

        Returns:
            Tuple of (is_valid, message).
        """
        if not api_key or len(api_key) < 10:
            return False, "API key is too short"

        # Check format
        if api_key.startswith("sk-or-"):
            return True, "API key format looks valid"

        return False, "Invalid API key format (should start with sk-or-)"


__all__ = ["LLMService", "ChatMessage", "CompletionResult", "LLMServiceError"]
