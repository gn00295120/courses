"""
Model utilities for dynamic model selection.

This module provides helper functions to dynamically select Claude models
based on aliases rather than hardcoded version IDs, protecting against model
deprecation and version changes.
"""

from anthropic import Anthropic
from typing import Optional


def get_latest_model(client: Anthropic, model_alias: str) -> str:
    """
    Get the latest model ID for a given model alias.

    This function queries the Anthropic API to find the latest version of a model
    based on its alias (e.g., 'sonnet', 'haiku', 'opus'), protecting against
    hardcoded model IDs becoming deprecated.

    Args:
        client: Anthropic API client instance
        model_alias: Model alias ('sonnet', 'haiku', 'opus')

    Returns:
        Latest model ID (e.g., 'claude-3-5-sonnet-20241022')

    Raises:
        ValueError: If model_alias is not recognized

    Examples:
        >>> client = Anthropic()
        >>> model = get_latest_model(client, 'sonnet')
        >>> print(model)
        'claude-3-5-sonnet-20241022'

        >>> model = get_latest_model(client, 'haiku')
        >>> print(model)
        'claude-3-haiku-20240307'
    """
    # Map of aliases to model name patterns
    alias_patterns = {
        'sonnet': 'sonnet',
        'haiku': 'haiku',
        'opus': 'opus',
        'sonnet-large': 'sonnet',
        'sonnet-small': 'haiku',  # Alias for consistency
    }

    if model_alias not in alias_patterns:
        raise ValueError(
            f"Unknown model alias: '{model_alias}'. "
            f"Valid options: {list(alias_patterns.keys())}"
        )

    pattern = alias_patterns[model_alias]

    # Get all models from API
    models = client.models.list()

    # Filter models matching the pattern
    matching_models = [
        m.id for m in models.data
        if pattern in m.id
    ]

    if not matching_models:
        raise ValueError(
            f"No models found matching alias '{model_alias}' (pattern: '{pattern}')"
        )

    # Sort by ID (which includes date) to get the latest
    # Models are named like claude-3-5-sonnet-YYYYMMDD, so sorting by name
    # in reverse order gives us the most recent
    latest_model = sorted(matching_models, reverse=True)[0]

    return latest_model


def get_model(client: Anthropic, model_alias: str) -> str:
    """
    Convenience function to get a model by alias.

    This is a shorter alias for get_latest_model().

    Args:
        client: Anthropic API client instance
        model_alias: Model alias ('sonnet', 'haiku', 'opus')

    Returns:
        Latest model ID

    Examples:
        >>> client = Anthropic()
        >>> model = get_model(client, 'sonnet')
        >>> response = client.messages.create(model=model, ...)
    """
    return get_latest_model(client, model_alias)


# Predefined model selection shortcuts
class ModelSelector:
    """
    Convenience class for model selection with predefined shortcuts.

    Examples:
        >>> client = Anthropic()
        >>> selector = ModelSelector(client)
        >>>
        >>> # Using shortcuts
        >>> model = selector.sonnet()
        >>> model = selector.haiku()
        >>> model = selector.latest_large()
        >>> model = selector.latest_small()
    """

    def __init__(self, client: Anthropic):
        """
        Initialize ModelSelector with an Anthropic client.

        Args:
            client: Anthropic API client instance
        """
        self.client = client

    def sonnet(self) -> str:
        """Get the latest Sonnet model (most capable)."""
        return get_latest_model(self.client, 'sonnet')

    def haiku(self) -> str:
        """Get the latest Haiku model (fastest)."""
        return get_latest_model(self.client, 'haiku')

    def opus(self) -> str:
        """Get the latest Opus model (if available)."""
        return get_latest_model(self.client, 'opus')

    def latest_large(self) -> str:
        """Get the latest large model (Sonnet)."""
        return self.sonnet()

    def latest_small(self) -> str:
        """Get the latest small model (Haiku)."""
        return self.haiku()


# Convenience instance for quick access
# Usage:
# from model_utils import selector
# model = selector.sonnet()
# _selector = None  # Will be initialized on first use
#
# def selector():
#     global _selector
#     if _selector is None:
#         from anthropic import Anthropic
#         _selector = ModelSelector(Anthropic())
#     return _selector
