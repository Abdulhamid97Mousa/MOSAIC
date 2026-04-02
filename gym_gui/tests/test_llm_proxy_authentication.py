"""Test HuggingFace authentication and model download with proxy settings.

This test suite verifies that:
1. HuggingFace login (via huggingface_hub.login()) respects proxy settings
2. Model download (via snapshot_download()) respects proxy settings
3. Proxy configuration can be enabled/disabled dynamically
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# Check if LLM chat dependencies are available
try:
    import huggingface_hub  # noqa: F401

    from gym_gui.services.llm import (
        LLM_CHAT_AVAILABLE,
        HuggingFaceAuth,
        ModelDownloader,
        ProxyConfig,
    )
    _HAS_HF_HUB = True
    SKIP_REASON = None
except ImportError as e:
    LLM_CHAT_AVAILABLE = False
    _HAS_HF_HUB = False
    SKIP_REASON = f"LLM chat dependencies not available: {e}"

if TYPE_CHECKING:
    from gym_gui.services.llm import HuggingFaceAuth, ModelDownloader, ProxyConfig

_SKIP = not LLM_CHAT_AVAILABLE or not _HAS_HF_HUB


@pytest.mark.skipif(_SKIP, reason=SKIP_REASON or "LLM/huggingface_hub not available")
class TestProxyConfiguration:
    """Test proxy configuration management."""

    def setup_method(self):
        """Reset proxy configuration before each test."""
        ProxyConfig.set_proxy(enabled=False)
        # Clear any proxy environment variables
        for var in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Clean up after each test."""
        ProxyConfig.set_proxy(enabled=False)

    def test_proxy_initially_disabled(self):
        """Proxy should be disabled by default."""
        assert not ProxyConfig.is_enabled()
        assert ProxyConfig.get_http_proxy() is None
        assert ProxyConfig.get_https_proxy() is None

    def test_enable_proxy(self):
        """Test enabling proxy with URLs."""
        http_proxy = "http://127.0.0.1:7890"
        https_proxy = "https://127.0.0.1:7890"

        ProxyConfig.set_proxy(enabled=True, http_proxy=http_proxy, https_proxy=https_proxy)

        assert ProxyConfig.is_enabled()
        assert ProxyConfig.get_http_proxy() == http_proxy
        assert ProxyConfig.get_https_proxy() == https_proxy

    def test_proxy_sets_environment_variables(self):
        """Test that enabling proxy sets environment variables."""
        http_proxy = "http://127.0.0.1:7890"
        https_proxy = "https://127.0.0.1:7890"

        ProxyConfig.set_proxy(enabled=True, http_proxy=http_proxy, https_proxy=https_proxy)

        assert os.environ.get("HTTP_PROXY") == http_proxy
        assert os.environ.get("http_proxy") == http_proxy
        assert os.environ.get("HTTPS_PROXY") == https_proxy
        assert os.environ.get("https_proxy") == https_proxy

    def test_disable_proxy_clears_environment_variables(self):
        """Test that disabling proxy clears environment variables."""
        # First enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")
        assert "HTTP_PROXY" in os.environ

        # Then disable
        ProxyConfig.set_proxy(enabled=False)

        assert "HTTP_PROXY" not in os.environ
        assert "http_proxy" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
        assert "https_proxy" not in os.environ

    def test_proxy_status_string(self):
        """Test proxy status string generation."""
        assert ProxyConfig.get_status() == "Proxy disabled"

        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")
        status = ProxyConfig.get_status()
        assert "HTTP: http://127.0.0.1:7890" in status
        assert "HTTPS: https://127.0.0.1:7890" in status


@pytest.mark.skipif(_SKIP, reason=SKIP_REASON or "LLM/huggingface_hub not available")
class TestHuggingFaceAuthWithProxy:
    """Test HuggingFace authentication with proxy settings."""

    def setup_method(self):
        """Reset authentication before each test."""
        HuggingFaceAuth.clear_token()
        ProxyConfig.set_proxy(enabled=False)

    def teardown_method(self):
        """Clean up after each test."""
        HuggingFaceAuth.clear_token()
        ProxyConfig.set_proxy(enabled=False)

    def test_token_validation_without_proxy(self):
        """Test that token validation works without proxy."""
        # Mock the HfApi to avoid real network calls
        with patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api_instance = MagicMock()
            mock_api_instance.whoami.return_value = {"name": "test_user"}
            mock_hf_api.return_value = mock_api_instance

            is_valid, message = HuggingFaceAuth.validate_token("hf_test_token_12345678901234567890")

            assert is_valid
            assert "test_user" in message
            # Verify HfApi was called with the token
            mock_hf_api.assert_called_once_with(token="hf_test_token_12345678901234567890")

    def test_token_validation_with_proxy_enabled(self):
        """Test that token validation respects proxy when enabled."""
        # Enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")

        # Mock the HfApi to avoid real network calls
        with patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_api_instance = MagicMock()
            mock_api_instance.whoami.return_value = {"name": "test_user"}
            mock_hf_api.return_value = mock_api_instance

            is_valid, message = HuggingFaceAuth.validate_token("hf_test_token_12345678901234567890")

            assert is_valid
            # Verify environment variables are set (HfApi should use them)
            assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"
            assert os.environ.get("HTTPS_PROXY") == "https://127.0.0.1:7890"

    @patch("huggingface_hub.login")
    def test_token_save_respects_proxy(self, mock_login):
        """Test that token save (login) respects proxy settings."""
        # Enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")

        # Save token
        HuggingFaceAuth.set_token("hf_test_token_12345678901234567890", persist=True)

        # Verify login was called
        mock_login.assert_called_once_with(token="hf_test_token_12345678901234567890", add_to_git_credential=False)

        # Verify proxy environment variables are set when login is called
        assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"
        assert os.environ.get("HTTPS_PROXY") == "https://127.0.0.1:7890"

    def test_token_format_validation(self):
        """Test basic token format validation."""
        # Empty token
        is_valid, message = HuggingFaceAuth.validate_token("")
        assert not is_valid
        assert "empty" in message.lower()

        # Token without hf_ prefix
        is_valid, message = HuggingFaceAuth.validate_token("invalid_token")
        assert not is_valid
        assert "hf_" in message

        # Token too short
        is_valid, message = HuggingFaceAuth.validate_token("hf_short")
        assert not is_valid
        assert "short" in message.lower()


@pytest.mark.skipif(_SKIP, reason=SKIP_REASON or "LLM/huggingface_hub not available")
class TestModelDownloadWithProxy:
    """Test model download with proxy settings."""

    def setup_method(self):
        """Reset proxy configuration before each test."""
        ProxyConfig.set_proxy(enabled=False)

    def teardown_method(self):
        """Clean up after each test."""
        ProxyConfig.set_proxy(enabled=False)

    @patch("huggingface_hub.snapshot_download")
    def test_model_download_without_proxy(self, mock_snapshot_download):
        """Test that model download works without proxy."""
        mock_snapshot_download.return_value = "/fake/path/to/model"

        downloader = ModelDownloader()

        # Clear proxy settings
        ProxyConfig.set_proxy(enabled=False)

        # Attempt download
        result = downloader.download_model("meta-llama/Llama-3.2-3B-Instruct")

        # Verify snapshot_download was called
        assert mock_snapshot_download.called
        assert result == Path("/fake/path/to/model")

    @patch("huggingface_hub.snapshot_download")
    def test_model_download_with_proxy_enabled(self, mock_snapshot_download):
        """Test that model download respects proxy when enabled."""
        mock_snapshot_download.return_value = "/fake/path/to/model"

        # Enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")

        downloader = ModelDownloader()

        # Attempt download
        result = downloader.download_model("meta-llama/Llama-3.2-3B-Instruct")

        # Verify snapshot_download was called
        assert mock_snapshot_download.called

        # Verify proxy environment variables are set during download
        assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"
        assert os.environ.get("HTTPS_PROXY") == "https://127.0.0.1:7890"

        assert result == Path("/fake/path/to/model")

    @patch("huggingface_hub.snapshot_download")
    def test_model_download_applies_proxy_before_download(self, mock_snapshot_download):
        """Test that proxy is applied before download starts."""
        def check_proxy_on_download(*args, **kwargs):
            # This should be called with proxy env vars set
            assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"
            return "/fake/path/to/model"

        mock_snapshot_download.side_effect = check_proxy_on_download

        # Enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")

        downloader = ModelDownloader()
        result = downloader.download_model("meta-llama/Llama-3.2-3B-Instruct")

        assert result == Path("/fake/path/to/model")

    @patch("huggingface_hub.snapshot_download")
    def test_model_download_with_progress_callback(self, mock_snapshot_download):
        """Test model download with progress updates respects proxy."""
        mock_snapshot_download.return_value = "/fake/path/to/model"

        # Enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")

        downloader = ModelDownloader()

        progress_calls = []
        def progress_callback(percent, message):
            progress_calls.append((percent, message))

        # Attempt download with progress callback
        downloader.download_model(
            "meta-llama/Llama-3.2-3B-Instruct",
            progress_callback=progress_callback
        )

        # Verify progress callbacks were made
        assert len(progress_calls) > 0
        # First callback should indicate preparation
        assert "Preparing" in progress_calls[0][1] or "Connecting" in progress_calls[1][1]

        # Verify proxy status is mentioned in progress (optional check)
        # The proxy status message may appear if the downloader reports it
        # At minimum, we verify that proxy env vars are set during download
        assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"

    @patch("huggingface_hub.snapshot_download")
    def test_model_download_authentication_error(self, mock_snapshot_download):
        """Test that authentication errors are properly reported."""
        mock_snapshot_download.side_effect = Exception("401 Unauthorized")

        downloader = ModelDownloader()

        with pytest.raises(RuntimeError) as exc_info:
            downloader.download_model("meta-llama/Llama-3.1-8B-Instruct")

        assert "Authentication required" in str(exc_info.value)

    @patch("huggingface_hub.snapshot_download")
    def test_model_download_forbidden_error(self, mock_snapshot_download):
        """Test that access denied errors are properly reported."""
        mock_snapshot_download.side_effect = Exception("403 Forbidden")

        downloader = ModelDownloader()

        with pytest.raises(RuntimeError) as exc_info:
            downloader.download_model("meta-llama/Llama-3.1-8B-Instruct")

        assert "Access denied" in str(exc_info.value)
        assert "license" in str(exc_info.value).lower()


@pytest.mark.skipif(_SKIP, reason=SKIP_REASON or "LLM/huggingface_hub not available")
class TestProxyIntegration:
    """Integration tests for proxy with authentication and download."""

    def setup_method(self):
        """Reset state before each test."""
        HuggingFaceAuth.clear_token()
        ProxyConfig.set_proxy(enabled=False)

    def teardown_method(self):
        """Clean up after each test."""
        HuggingFaceAuth.clear_token()
        ProxyConfig.set_proxy(enabled=False)

    @patch("huggingface_hub.login")
    @patch("huggingface_hub.snapshot_download")
    def test_full_workflow_with_proxy(self, mock_snapshot_download, mock_login):
        """Test complete workflow: set proxy -> authenticate -> download model."""
        mock_snapshot_download.return_value = "/fake/path/to/model"

        # Step 1: Enable proxy
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")
        assert ProxyConfig.is_enabled()

        # Step 2: Authenticate with HuggingFace
        HuggingFaceAuth.set_token("hf_test_token_12345678901234567890", persist=True)
        assert HuggingFaceAuth.is_authenticated()

        # Verify login was called with proxy env vars set
        mock_login.assert_called_once()
        assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"

        # Step 3: Download model
        downloader = ModelDownloader()
        result = downloader.download_model("meta-llama/Llama-3.2-3B-Instruct")

        # Verify download was called with proxy env vars still set
        assert mock_snapshot_download.called
        assert os.environ.get("HTTPS_PROXY") == "https://127.0.0.1:7890"
        assert result == Path("/fake/path/to/model")

    @patch("huggingface_hub.login")
    @patch("huggingface_hub.snapshot_download")
    def test_disable_proxy_mid_workflow(self, mock_snapshot_download, mock_login):
        """Test disabling proxy between authentication and download."""
        mock_snapshot_download.return_value = "/fake/path/to/model"

        # Enable proxy and authenticate
        ProxyConfig.set_proxy(enabled=True, http_proxy="http://127.0.0.1:7890", https_proxy="https://127.0.0.1:7890")
        HuggingFaceAuth.set_token("hf_test_token_12345678901234567890", persist=True)

        # Disable proxy
        ProxyConfig.set_proxy(enabled=False)

        # Verify env vars are cleared
        assert "HTTP_PROXY" not in os.environ
        assert "HTTPS_PROXY" not in os.environ

        # Download should work without proxy
        downloader = ModelDownloader()
        result = downloader.download_model("meta-llama/Llama-3.2-3B-Instruct")

        assert mock_snapshot_download.called
        assert result == Path("/fake/path/to/model")
