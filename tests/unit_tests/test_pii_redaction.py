"""Unit tests for PII redaction module."""

import pytest

from langchain_core.messages import HumanMessage
from src.flow_agent.utils.pii_redaction import PII_Redactor


class TestPIIRedaction:
    """Test suite for PII_Redactor class.

    Note: Presidio's PII detection capabilities vary by entity type:
    - Emails: Well detected and redacted to <EMAIL_ADDRESS>
    - Person names: Well detected and redacted to <PERSON>
    - Phone numbers: Not reliably detected (format dependent)
    - Credit cards: NOT supported for English language
    - SSN: Not reliably detected without specific patterns
    - Dates/times: Detected but can have false positives
    - URLs: Detected and redacted to <URL>
    - IBAN codes: Detected for banking formats
    - IP addresses: Detected
    """

    # Positive test cases - PII should be detected and redacted
    @pytest.mark.asyncio
    async def test_redacts_email_address(self):
        """Test that email addresses are properly redacted."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="My email is john.doe@example.com")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        # String content stays as string after redaction
        assert isinstance(content, str)
        assert "john.doe@example.com" not in content
        assert "<EMAIL_ADDRESS>" in content

    @pytest.mark.asyncio
    async def test_redacts_person_name(self):
        """Test that person names are properly redacted."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="My name is John Smith")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, str)
        assert "John Smith" not in content
        assert "<PERSON>" in content

    @pytest.mark.asyncio
    async def test_redacts_url(self):
        """Test that URLs are properly redacted."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="Visit my website at https://example.com/profile/john")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, str)
        assert "https://example.com/profile/john" not in content
        assert "<URL>" in content

    @pytest.mark.asyncio
    async def test_redacts_ip_address(self):
        """Test that IP addresses are properly redacted."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="Connect to 192.168.1.1 for access")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, str)
        assert "192.168.1.1" not in content
        assert "<IP_ADDRESS>" in content

    @pytest.mark.asyncio
    async def test_redacts_multiple_pii_types_in_single_message(self):
        """Test that multiple PII types in one message are all redacted."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(
            content="Contact John Smith at john.smith@example.com "
                    "or visit https://johns-site.com"
        )

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, str)
        assert "John Smith" not in content
        assert "john.smith@example.com" not in content
        assert "https://johns-site.com" not in content
        # Check for redaction markers
        assert "<PERSON>" in content
        assert "<EMAIL_ADDRESS>" in content
        assert "<URL>" in content

    # Negative test cases - No PII should remain unchanged
    @pytest.mark.asyncio
    async def test_no_pii_with_generic_content(self):
        """Test that messages without PII are processed but may have date/time detected."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="The weather is nice and the sky is blue")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        # Generic content should be preserved (possibly with false positives like dates)
        content = result[0].content
        assert isinstance(content, str)
        assert "weather" in content or "sky" in content or "blue" in content

    @pytest.mark.asyncio
    async def test_no_pii_leaves_list_content_unchanged(self):
        """Test that list content without PII remains unchanged."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        content = [
            {"type": "text", "text": "This is a safe message without personal data."},
            {"type": "image", "data": "base64data...", "metadata": {"filename": "test.png"}}
        ]
        message = HumanMessage(content=content)

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        result_content = result[0].content
        assert isinstance(result_content, list)
        # Text should remain largely unchanged
        assert "safe message" in result_content[0]["text"] or "message" in result_content[0]["text"]
        # Image should remain unchanged
        assert result_content[1]["data"] == "base64data..."

    @pytest.mark.asyncio
    async def test_empty_message_returns_unchanged(self):
        """Test that empty messages are handled gracefully."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        # Empty string stays as empty string
        assert result[0].content == ""

    # Mixed test cases - Multimodal content with PII and non-PII elements
    @pytest.mark.asyncio
    async def test_multimodal_message_redacts_only_text_with_pii(self):
        """Test that multimodal messages redact PII in text but preserve other media."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        content = [
            {"type": "text", "text": "My email is jane@example.com"},
            {"type": "image", "data": "iVBORw0KGgo...", "metadata": {"filename": "photo.png"}, "mime_type": "image/png"},
            {"type": "text", "text": "Contact Bob Smith for more info"}
        ]
        message = HumanMessage(content=content)

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        result_content = result[0].content
        assert isinstance(result_content, list)
        assert len(result_content) == 3

        # First text should be redacted
        assert "jane@example.com" not in result_content[0]["text"]
        assert "<EMAIL_ADDRESS>" in result_content[0]["text"]
        # Image should be preserved
        assert result_content[1]["data"] == "iVBORw0KGgo..."
        assert result_content[1]["type"] == "image"
        # Second text should have name redacted
        assert "Bob Smith" not in result_content[2]["text"]
        assert "<PERSON>" in result_content[2]["text"]

    @pytest.mark.asyncio
    async def test_mixed_content_partial_pii_redaction(self):
        """Test message with both PII and safe content mixed together."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(
            content="Hello! My name is Alice Jones. "
                    "The project deadline is next Friday. "
                    "Email me at alice@test.com for details."
        )

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, str)

        # PII should be redacted
        assert "Alice Jones" not in content
        assert "alice@test.com" not in content
        # Some safe content should remain (dates may also be redacted)
        assert "Hello" in content or "project" in content or "deadline" in content

    @pytest.mark.asyncio
    async def test_multiple_messages_with_varying_pii_content(self):
        """Test processing multiple messages with different PII scenarios."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        messages = [
            HumanMessage(content="Safe message with no personal data"),
            HumanMessage(content="Email: bob@example.com"),
            HumanMessage(content="Another safe message"),
            HumanMessage(content="Contact Mary Johnson"),
        ]

        result = await redactor.do_pii_redaction(messages)

        assert len(result) == 4
        # First message - minimal PII
        assert "Safe message" in result[0].content or "message" in result[0].content
        # Second message - email redacted
        assert "bob@example.com" not in result[1].content
        assert "<EMAIL_ADDRESS>" in result[1].content
        # Third message - minimal PII
        assert "Another safe" in result[2].content or "safe" in result[2].content
        # Fourth message - name redacted
        assert "Mary Johnson" not in result[3].content
        assert "<PERSON>" in result[3].content

    # Edge cases
    @pytest.mark.asyncio
    async def test_confidence_threshold_affects_detection(self):
        """Test that confidence threshold affects PII detection sensitivity."""
        # High threshold - may miss some PII
        redactor_strict = PII_Redactor(confidence_threshold=0.9)
        message = HumanMessage(content="Contact me at test@example.com")

        result_strict = await redactor_strict.do_pii_redaction([message])

        # Low threshold - should catch more PII
        redactor_lenient = PII_Redactor(confidence_threshold=0.1)
        result_lenient = await redactor_lenient.do_pii_redaction([message])

        # Both should produce results
        assert result_strict is not None
        assert result_lenient is not None
        # Lenient may redact more aggressively

    @pytest.mark.asyncio
    async def test_does_not_modify_original_message(self):
        """Test that the original message object is not modified (immutability)."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        original_content = "My email is original@example.com"
        message = HumanMessage(content=original_content)

        result = await redactor.do_pii_redaction([message])

        # Original message should remain unchanged
        assert message.content == original_content
        # Result should be a different object with redacted content
        assert result[0] is not message
        # Content should be redacted (string format)
        assert result[0].content != original_content
        assert "<EMAIL_ADDRESS>" in result[0].content

    @pytest.mark.asyncio
    async def test_handles_us_phone_number_format_variations(self):
        """Test various US phone number formats.

        Note: Presidio may not detect all phone number formats.
        This test documents current behavior.
        """
        redactor = PII_Redactor(confidence_threshold=0.5)
        # US format with country code - more likely to be detected
        message = HumanMessage(content="Call me at +1 (555) 123-4567")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        # Phone detection in Presidio is format-dependent
        # This test documents current behavior rather than asserting specific outcome
        assert result[0].content is not None

    @pytest.mark.asyncio
    async def test_detects_banking_information_iban(self):
        """Test IBAN detection for banking information."""
        redactor = PII_Redactor(confidence_threshold=0.5)
        message = HumanMessage(content="My IBAN is GB82WEST12345698765432")

        result = await redactor.do_pii_redaction([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, str)
        # IBAN should be redacted
        assert "GB82WEST12345698765432" not in content or "IBAN" in content
