"""Unit tests for ClaudeAPIClient."""

from unittest.mock import patch

from claudecode.claude_api_client import ClaudeAPIClient


class TestValidateApiAccess:
    """Test API access validation."""

    def test_validates_with_configured_model(self):
        """The probe must use the client's own model, not a hardcoded one that can be retired."""
        with patch('claudecode.claude_api_client.Anthropic') as mock_anthropic:
            client = ClaudeAPIClient(model='claude-sonnet-5', api_key='test-key')
            valid, error = client.validate_api_access()

        assert valid is True
        assert error == ""
        create = mock_anthropic.return_value.messages.create
        assert create.call_args.kwargs['model'] == 'claude-sonnet-5'

    def test_reports_failure(self):
        """A failed probe returns the error instead of raising."""
        with patch('claudecode.claude_api_client.Anthropic') as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception('model not found')
            client = ClaudeAPIClient(model='claude-sonnet-5', api_key='test-key')
            valid, error = client.validate_api_access()

        assert valid is False
        assert 'model not found' in error
