"""LLM service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from gym_gui.services.llm.models import LLMProvider, ModelIdentity


@dataclass
class LLMConfig:
    """LLM service configuration.

    Supports both environment variable and UI-based API key configuration.
    OpenRouter is the primary provider.

    Attributes:
        enabled: Whether LLM chat is enabled.
        active_provider: Currently active provider.
        openrouter_api_key: API key (from UI or env var).
        vllm_base_url: Base URL for local vLLM server.
        preferred_models: Curated list of models to show in UI.
        max_history_messages: Maximum messages to keep in chat history.
        default_max_tokens: Default token limit for responses.
        default_temperature: Default sampling temperature.
        request_timeout_seconds: HTTP request timeout.
    """

    enabled: bool = True
    active_provider: LLMProvider = LLMProvider.OPENROUTER

    # OpenRouter settings (primary)
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # vLLM settings (local fallback)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"

    # Zhipu AI settings (GLM models)
    # z.ai provides Anthropic-compatible API
    # Environment variable: ZHIPU_API_KEY or ZAI_API_KEY
    zhipu_api_key: Optional[str] = None
    zhipu_base_url: str = "https://api.z.ai/api/anthropic"  # Anthropic-compatible endpoint

    # Curated model list (not auto-discovery)
    preferred_models: List[ModelIdentity] = field(default_factory=lambda: [
        # Zhipu AI models (GLM series via z.ai - Anthropic-compatible API)
        ModelIdentity(LLMProvider.ZHIPU, "glm-5", "GLM-5 (Zhipu AI, newest flagship)"),
        ModelIdentity(LLMProvider.ZHIPU, "glm-4.7", "GLM-4.7 (Zhipu AI, reasoning)"),
        ModelIdentity(LLMProvider.ZHIPU, "glm-4.6", "GLM-4.6 (Zhipu AI, 200K context)"),
        ModelIdentity(LLMProvider.ZHIPU, "glm-4.5", "GLM-4.5 (Zhipu AI, 128K context)"),
        ModelIdentity(LLMProvider.ZHIPU, "glm-4.5-flash", "GLM-4.5 Flash (Zhipu AI, FREE)"),
        ModelIdentity(LLMProvider.ZHIPU, "glm-4.5-air", "GLM-4.5 Air (Zhipu AI, lightweight)"),
        # OpenRouter cloud models (curated, not the full 1000+ catalog)
        ModelIdentity(LLMProvider.OPENROUTER, "openai/gpt-4o-mini", "GPT-4o Mini"),
        ModelIdentity(LLMProvider.OPENROUTER, "openai/gpt-4o", "GPT-4o"),
        ModelIdentity(LLMProvider.OPENROUTER, "anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ModelIdentity(LLMProvider.OPENROUTER, "anthropic/claude-3-haiku", "Claude 3 Haiku"),
        ModelIdentity(LLMProvider.OPENROUTER, "google/gemini-pro-1.5", "Gemini Pro 1.5"),
        ModelIdentity(LLMProvider.OPENROUTER, "meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B"),
        ModelIdentity(LLMProvider.OPENROUTER, "meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B"),
        ModelIdentity(LLMProvider.OPENROUTER, "mistralai/mistral-7b-instruct", "Mistral 7B"),
        # vLLM local models (user must configure)
        # Use HuggingFace model IDs - vLLM will download automatically
        # Models with permissive licenses (no gating) - recommended for easy setup
        ModelIdentity(LLMProvider.VLLM, "Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 1.5B (local, fast)"),
        ModelIdentity(LLMProvider.VLLM, "Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B (local)"),
        ModelIdentity(LLMProvider.VLLM, "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B (local)"),
        ModelIdentity(LLMProvider.VLLM, "microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini 4K (local)"),
        ModelIdentity(LLMProvider.VLLM, "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen 2.5 Coder 7B (local)"),
        ModelIdentity(LLMProvider.VLLM, "deepseek-ai/deepseek-coder-6.7b-instruct", "DeepSeek Coder 6.7B (local)"),
        # Gated models (require HuggingFace license approval)
        ModelIdentity(LLMProvider.VLLM, "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B (local, gated)"),
        ModelIdentity(LLMProvider.VLLM, "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B (local, gated)"),
        ModelIdentity(LLMProvider.VLLM, "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B (local, gated)"),
    ])

    # Chat settings
    max_history_messages: int = 20
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    request_timeout_seconds: int = 60

    def get_models_for_provider(self, provider: LLMProvider) -> List[ModelIdentity]:
        """Get preferred models filtered by provider."""
        return [m for m in self.preferred_models if m.provider == provider]

    def get_active_models(self) -> List[ModelIdentity]:
        """Get models for the currently active provider."""
        return self.get_models_for_provider(self.active_provider)

    def get_api_key(self) -> Optional[str]:
        """Get the API key for the active provider.

        Checks UI-provided key first, then environment variable.
        """
        if self.active_provider == LLMProvider.OPENROUTER:
            # UI key takes precedence over env var
            if self.openrouter_api_key:
                return self.openrouter_api_key
            return os.getenv("OPENROUTER_API_KEY")
        elif self.active_provider == LLMProvider.VLLM:
            return self.vllm_api_key
        elif self.active_provider == LLMProvider.ZHIPU:
            if self.zhipu_api_key:
                return self.zhipu_api_key
            return os.getenv("ZHIPU_API_KEY") or os.getenv("ZAI_API_KEY")
        return None

    def set_api_key(self, api_key: str) -> None:
        """Set the API key for the active provider.

        Args:
            api_key: The API key to set.
        """
        if self.active_provider == LLMProvider.OPENROUTER:
            self.openrouter_api_key = api_key
        elif self.active_provider == LLMProvider.VLLM:
            self.vllm_api_key = api_key
        elif self.active_provider == LLMProvider.ZHIPU:
            self.zhipu_api_key = api_key

    def has_valid_api_key(self) -> bool:
        """Check if a valid API key is configured."""
        api_key = self.get_api_key()
        if api_key is None or api_key == "":
            return False
        if self.active_provider == LLMProvider.VLLM:
            return True  # vLLM accepts "EMPTY"
        return len(api_key) > 10  # OpenRouter/Zhipu keys are longer

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Create config from environment variables.

        Environment variables:
            OPENROUTER_API_KEY: OpenRouter API key
            VLLM_BASE_URL: vLLM server URL (default: http://localhost:8000/v1)
            ZHIPU_API_KEY / ZAI_API_KEY: Zhipu AI API key (z.ai)
            ZHIPU_BASE_URL: Zhipu AI API URL (default: https://api.z.ai/api/anthropic)
        """
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            zhipu_api_key=os.getenv("ZHIPU_API_KEY") or os.getenv("ZAI_API_KEY"),
            zhipu_base_url=os.getenv("ZHIPU_BASE_URL", "https://api.z.ai/api/anthropic"),
        )


__all__ = ["LLMConfig"]
