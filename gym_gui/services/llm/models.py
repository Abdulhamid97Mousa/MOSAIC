"""LLM model identity and provider definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENROUTER = "openrouter"
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ZHIPU = "zhipu"


@dataclass(frozen=True)
class ModelIdentity:
    """Unambiguous model identity with provider context.

    Uses provider-namespaced IDs to avoid model name collisions
    across different providers.

    Examples:
        >>> model = ModelIdentity(LLMProvider.OPENROUTER, "openai/gpt-4o-mini", "GPT-4o Mini")
        >>> model.namespaced_id
        'openrouter::openai/gpt-4o-mini'

        >>> ModelIdentity.from_namespaced("vllm::mistral-7b-instruct")
        ModelIdentity(provider=<LLMProvider.VLLM>, model_id='mistral-7b-instruct', ...)
    """

    provider: LLMProvider
    model_id: str
    display_name: Optional[str] = None

    @property
    def namespaced_id(self) -> str:
        """Provider-namespaced identifier.

        Format: 'provider::model_id'
        e.g., 'openrouter::openai/gpt-4o-mini' or 'vllm::mistral-7b-instruct'
        """
        return f"{self.provider.value}::{self.model_id}"

    @classmethod
    def from_namespaced(cls, namespaced: str) -> "ModelIdentity":
        """Parse 'provider::model_id' format.

        Args:
            namespaced: Provider-namespaced model ID string.

        Returns:
            ModelIdentity instance.

        Raises:
            ValueError: If format is invalid or provider unknown.
        """
        if "::" not in namespaced:
            raise ValueError(f"Invalid namespaced ID (missing '::'): {namespaced}")

        provider_str, model_id = namespaced.split("::", 1)

        try:
            provider = LLMProvider(provider_str)
        except ValueError as e:
            raise ValueError(f"Unknown provider '{provider_str}' in: {namespaced}") from e

        return cls(provider=provider, model_id=model_id)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.display_name:
            return self.display_name
        return f"{self.provider.value}: {self.model_id}"


__all__ = ["LLMProvider", "ModelIdentity"]
