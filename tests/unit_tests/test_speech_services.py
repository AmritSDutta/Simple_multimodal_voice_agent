"""Unit tests for speech services.

Tests for:
- Abstract interface compliance
- SarvamAI implementation (with mocked API calls)
- Factory function behavior
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.flow_agent.speech.factory import get_speech_service, list_available_providers
from src.flow_agent.speech.interface import SpeechService
from src.flow_agent.speech.providers.gen_ai import GenAiSpeechService
from src.flow_agent.speech.providers.open_ai import OpenAiSpeechService
from src.flow_agent.speech.providers.sarvam_ai import SarvamAiSpeechService


class TestSpeechServiceInterface:
    """Test the abstract interface compliance."""

    def test_sarvam_service_is_speech_service(self):
        """SarvamAiSpeechService should be a SpeechService."""
        # Mock the API key for instantiation
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            service = SarvamAiSpeechService()
            assert isinstance(service, SpeechService)

    def test_speech_service_has_provider_name(self):
        """SpeechService should have a provider_name property."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            service = SarvamAiSpeechService()
            assert hasattr(service, "provider_name")
            assert service.provider_name == "sarvam"

    def test_speech_service_has_abstract_methods(self):
        """SpeechService should have abstract methods defined."""
        assert hasattr(SpeechService, "speech_to_text")
        assert hasattr(SpeechService, "text_to_speech")
        assert hasattr(SpeechService, "provider_name")


class TestSarvamAiSpeechService:
    """Test SarvamAI implementation (with mocked API calls)."""

    @pytest.fixture
    def mock_sarvam_client(self):
        """Create a mock SarvamAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def sarvam_service(self, mock_sarvam_client):
        """Create SarvamAiSpeechService with mocked client."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            service = SarvamAiSpeechService()
            service._client = mock_sarvam_client
            return service

    def test_init_requires_api_key(self):
        """Initialization should raise ValueError if API key is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SARVAM_API_KEY"):
                SarvamAiSpeechService(api_key=None)

    def test_init_with_explicit_api_key(self):
        """Initialization should work with explicit API key."""
        service = SarvamAiSpeechService(api_key="explicit-key")
        assert service._api_key == "explicit-key"

    def test_init_with_default_params(self):
        """Initialization should use default parameters."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            service = SarvamAiSpeechService()
            assert service._stt_model == "saaras:v3"
            assert service._tts_model == "bulbul:v3"
            assert service._language == "en-IN"
            assert service._speaker == "shubh"
            assert service._tts_pace == 1.1
            assert service._tts_sample_rate == 22050

    def test_init_with_custom_params(self):
        """Initialization should accept custom parameters."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            service = SarvamAiSpeechService(
                stt_model="custom-stt",
                tts_model="custom-tts",
                language="hi-IN",
                speaker="custom-speaker",
                tts_pace=1.5,
                tts_sample_rate=44100,
            )
            assert service._stt_model == "custom-stt"
            assert service._tts_model == "custom-tts"
            assert service._language == "hi-IN"
            assert service._speaker == "custom-speaker"
            assert service._tts_pace == 1.5
            assert service._tts_sample_rate == 44100

    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, sarvam_service, mock_sarvam_client):
        """Test successful speech-to-text transcription."""
        # Mock the job object
        mock_job = MagicMock()
        mock_job.is_failed.return_value = False
        mock_job.wait_until_complete.return_value = MagicMock(error_message=None)

        # Mock job creation and file operations
        mock_sarvam_client.speech_to_text_job.create_job.return_value = mock_job

        # Mock file operations by patching tempfile and os
        with patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
             patch("tempfile.mkdtemp") as mock_mkdtemp, \
             patch("os.walk") as mock_walk, \
             patch("os.path.exists") as mock_exists, \
             patch("os.unlink"), \
             patch("os.rmdir"):

            # Setup temp file mock
            temp_file_obj = MagicMock()
            temp_file_obj.name = "/tmp/test_audio.webm"
            mock_temp_file.return_value.__enter__ = MagicMock(return_value=temp_file_obj)
            mock_temp_file.return_value.__exit__ = MagicMock(return_value=False)

            # Setup output directory mock
            mock_mkdtemp.return_value = "/tmp/output"

            # Setup os.walk to return transcript JSON
            mock_walk.return_value = [
                ("/tmp/output", [], ["transcript.json"])
            ]

            # Setup mock_exists to return False for cleanup
            mock_exists.return_value = False

            # Mock json.load to return transcript data
            with patch("builtins.open", MagicMock()), \
                 patch("json.load") as mock_json_load:

                mock_json_load.return_value = {"transcript": "Hello world"}

                # Call the method
                result = await sarvam_service.speech_to_text(b"fake audio data")

                # Verify result
                assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_speech_to_text_job_failure(self, sarvam_service, mock_sarvam_client):
        """Test speech-to-text when the job fails."""
        # Mock the job object that fails
        mock_job = MagicMock()
        mock_job.is_failed.return_value = True
        mock_job.wait_until_complete.return_value = MagicMock(
            error_message="Processing failed"
        )

        mock_sarvam_client.speech_to_text_job.create_job.return_value = mock_job

        with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
            temp_file_obj = MagicMock()
            temp_file_obj.name = "/tmp/test_audio.webm"
            mock_temp_file.return_value.__enter__ = MagicMock(return_value=temp_file_obj)
            mock_temp_file.return_value.__exit__ = MagicMock(return_value=False)

            # Call the method
            result = await sarvam_service.speech_to_text(b"fake audio data")

            # Verify None is returned on failure
            assert result is None

    @pytest.mark.asyncio
    async def test_speech_to_text_no_transcript(self, sarvam_service, mock_sarvam_client):
        """Test speech-to-text when no transcript is found."""
        mock_job = MagicMock()
        mock_job.is_failed.return_value = False
        mock_job.wait_until_complete.return_value = MagicMock(error_message=None)

        mock_sarvam_client.speech_to_text_job.create_job.return_value = mock_job

        with patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
             patch("tempfile.mkdtemp") as mock_mkdtemp, \
             patch("os.walk") as mock_walk, \
             patch("os.path.exists") as mock_exists:

            temp_file_obj = MagicMock()
            temp_file_obj.name = "/tmp/test_audio.webm"
            mock_temp_file.return_value.__enter__ = MagicMock(return_value=temp_file_obj)
            mock_temp_file.return_value.__exit__ = MagicMock(return_value=False)

            mock_mkdtemp.return_value = "/tmp/output"
            mock_walk.return_value = [("/tmp/output", [], [])]
            mock_exists.return_value = False

            result = await sarvam_service.speech_to_text(b"fake audio data")

            # Verify None is returned when no transcript found
            assert result is None

    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, sarvam_service, mock_sarvam_client):
        """Test successful text-to-speech conversion."""
        # Mock the TTS response
        mock_response = MagicMock()
        mock_response.audios = ["base64-audio-data-1", "base64-audio-data-2"]

        mock_sarvam_client.text_to_speech.convert.return_value = mock_response

        # Call the method
        result = await sarvam_service.text_to_speech("Hello world")

        # Verify result
        assert result == ["base64-audio-data-1", "base64-audio-data-2"]

        # Verify the API was called with correct parameters
        mock_sarvam_client.text_to_speech.convert.assert_called_once()
        call_args = mock_sarvam_client.text_to_speech.convert.call_args
        assert call_args[1]["text"] == "Hello world"
        assert call_args[1]["model"] == "bulbul:v3"  # TTS uses bulbul:v3 model

    @pytest.mark.asyncio
    async def test_text_to_speech_failure(self, sarvam_service, mock_sarvam_client):
        """Test text-to-speech when API call fails."""
        # Mock exception
        mock_sarvam_client.text_to_speech.convert.side_effect = Exception("API Error")

        # Call the method
        result = await sarvam_service.text_to_speech("Hello world")

        # Verify None is returned on failure
        assert result is None

    def test_extract_transcript_from_output(self, sarvam_service):
        """Test transcript extraction from output JSON."""
        import tempfile
        import json

        # Create a temporary JSON file with transcript
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "transcript.json")
            with open(json_path, "w") as f:
                json.dump({"transcript": "Test transcript"}, f)

            # Extract transcript
            result = sarvam_service._extract_transcript_from_output(temp_dir)

            assert result == "Test transcript"

    def test_extract_transcript_from_segments(self, sarvam_service):
        """Test transcript extraction from segments format."""
        import tempfile
        import json

        # Create a temporary JSON file with segments
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "output.json")
            with open(json_path, "w") as f:
                json.dump({
                    "segments": [
                        {"text": "Hello"},
                        {"text": "world"}
                    ]
                }, f)

            # Extract transcript
            result = sarvam_service._extract_transcript_from_output(temp_dir)

            assert result == "Hello world"


class TestOpenAiSpeechService:
    """Test OpenAI implementation (with mocked API calls)."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def openai_service(self, mock_openai_client):
        """Create OpenAiSpeechService with mocked client."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            service = OpenAiSpeechService()
            service._client = mock_openai_client
            return service

    def test_init_requires_api_key(self):
        """Initialization should raise ValueError if API key is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAiSpeechService(api_key=None)

    def test_init_with_explicit_api_key(self):
        """Initialization should work with explicit API key."""
        service = OpenAiSpeechService(api_key="explicit-key")
        assert service._api_key == "explicit-key"

    def test_init_with_default_params(self):
        """Initialization should use default parameters."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            service = OpenAiSpeechService()
            assert service._stt_model == "whisper-1"
            assert service._tts_model == "gpt-4o-mini-tts"
            assert service._language == "en-IN"
            assert service._speaker == "coral"
            assert service._tts_pace == 1.1
            assert service._tts_sample_rate == 22050

    def test_init_with_custom_params(self):
        """Initialization should accept custom parameters."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            service = OpenAiSpeechService(
                stt_model="whisper-2",
                tts_model="tts-1",
                language="es-ES",
                speaker="alloy",
                tts_pace=1.5,
                tts_sample_rate=44100,
            )
            assert service._stt_model == "whisper-2"
            assert service._tts_model == "tts-1"
            assert service._language == "es-ES"
            assert service._speaker == "alloy"
            assert service._tts_pace == 1.5
            assert service._tts_sample_rate == 44100

    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, openai_service, mock_openai_client):
        """Test successful speech-to-text transcription."""
        # Mock the transcription response
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_openai_client.audio.transcriptions.create.return_value = mock_response

        # Mock file operations
        with patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
             patch("builtins.open", MagicMock()), \
             patch("os.path.exists") as mock_exists, \
             patch("os.unlink"):

            # Setup temp file mock
            temp_file_obj = MagicMock()
            temp_file_obj.name = "/tmp/test_audio.webm"
            mock_temp_file.return_value.__enter__ = MagicMock(return_value=temp_file_obj)
            mock_temp_file.return_value.__exit__ = MagicMock(return_value=False)
            mock_exists.return_value = True

            # Call the method
            result = await openai_service.speech_to_text(b"fake audio data")

            # Verify result
            assert result == "Hello world"

            # Verify the API was called with correct parameters
            mock_openai_client.audio.transcriptions.create.assert_called_once()
            call_args = mock_openai_client.audio.transcriptions.create.call_args
            assert call_args[1]["model"] == "whisper-1"
            assert call_args[1]["language"] == "en"  # Extracted from "en-IN"

    @pytest.mark.asyncio
    async def test_speech_to_text_failure(self, openai_service, mock_openai_client):
        """Test speech-to-text when API call fails."""
        # Mock exception
        mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")

        # Mock file operations
        with patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
             patch("os.path.exists") as mock_exists, \
             patch("os.unlink"):

            temp_file_obj = MagicMock()
            temp_file_obj.name = "/tmp/test_audio.webm"
            mock_temp_file.return_value.__enter__ = MagicMock(return_value=temp_file_obj)
            mock_temp_file.return_value.__exit__ = MagicMock(return_value=False)
            mock_exists.return_value = True

            # Call the method
            result = await openai_service.speech_to_text(b"fake audio data")

            # Verify None is returned on failure
            assert result is None

    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, openai_service, mock_openai_client):
        """Test successful text-to-speech conversion."""
        # Mock the TTS response
        mock_response = MagicMock()
        mock_response.content = b"fake audio bytes"
        mock_openai_client.audio.speech.create.return_value = mock_response

        # Call the method
        result = await openai_service.text_to_speech("Hello world")

        # Verify result is a list with base64 encoded audio
        assert result is not None
        assert len(result) == 1
        import base64
        decoded = base64.b64decode(result[0])
        assert decoded == b"fake audio bytes"

        # Verify the API was called with correct parameters
        mock_openai_client.audio.speech.create.assert_called_once()
        call_args = mock_openai_client.audio.speech.create.call_args
        assert call_args[1]["input"] == "Hello world"  # OpenAI uses "input" not "text"
        assert call_args[1]["model"] == "gpt-4o-mini-tts"
        assert call_args[1]["voice"] == "coral"

    @pytest.mark.asyncio
    async def test_text_to_speech_failure(self, openai_service, mock_openai_client):
        """Test text-to-speech when API call fails."""
        # Mock exception
        mock_openai_client.audio.speech.create.side_effect = Exception("API Error")

        # Call the method
        result = await openai_service.text_to_speech("Hello world")

        # Verify None is returned on failure
        assert result is None


class TestSpeechServiceFactory:
    """Test factory function."""

    def test_returns_sarvam_provider_default(self):
        """Factory should return SarvamAiSpeechService by default."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service()
                assert isinstance(service, SarvamAiSpeechService)
                assert service.provider_name == "sarvam"

            asyncio.run(_test())

    def test_returns_sarvam_provider_explicit(self):
        """Factory should return SarvamAiSpeechService when explicitly requested."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service("sarvam")
                assert isinstance(service, SarvamAiSpeechService)

            asyncio.run(_test())

    def test_normalizes_provider_name(self):
        """Factory should normalize provider names (handle aliases)."""
        with patch.dict(
            os.environ,
            {
                "SARVAM_API_KEY": "test-key",
                "OPENAI_API_KEY": "test-key",
                "GEMINI_API_KEY": "test-key",
            },
        ):
            import asyncio

            async def _test():
                # Test SarvamAI aliases
                for alias in ["sarvam", "Sarvam", "SARVAM", "sarvamai", "SarvamAI"]:
                    service = await get_speech_service(alias)
                    assert isinstance(service, SarvamAiSpeechService)

                # Test OpenAI aliases
                for alias in ["openai", "OpenAI", "OPENAI", "open_ai"]:
                    service = await get_speech_service(alias)
                    assert isinstance(service, OpenAiSpeechService)

                # Test GenAI/Gemini aliases
                for alias in ["gemini", "Gemini", "GEMINI", "genai", "gen_ai", "google"]:
                    service = await get_speech_service(alias)
                    assert isinstance(service, GenAiSpeechService)

            asyncio.run(_test())

    def test_with_unknown_provider_raises_error(self):
        """Factory should raise ValueError for unknown provider."""
        import asyncio

        async def _test():
            with pytest.raises(ValueError, match="Unknown speech provider"):
                await get_speech_service("unknown_provider")

        asyncio.run(_test())

    def test_list_available_providers(self):
        """Factory should list available providers."""
        providers = list_available_providers()
        assert isinstance(providers, list)
        assert "sarvam" in providers
        assert "openai" in providers
        assert "gemini" in providers

    def test_returns_openai_provider_explicit(self):
        """Factory should return OpenAiSpeechService when explicitly requested."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service("openai")
                assert isinstance(service, OpenAiSpeechService)
                assert service.provider_name == "openai"

            asyncio.run(_test())

    def test_returns_genai_provider_explicit(self):
        """Factory should return GenAiSpeechService when explicitly requested."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service("gemini")
                assert isinstance(service, GenAiSpeechService)
                assert service.provider_name == "gemini"

            asyncio.run(_test())


class TestSpeechServiceFactoryWithSettings:
    """Test factory with custom settings."""

    def test_uses_settings_defaults(self):
        """Factory should use settings for SarvamAI defaults."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            from src.flow_agent.configurations.config import settings

            import asyncio

            async def _test():
                service = await get_speech_service()
                assert isinstance(service, SarvamAiSpeechService)
                # Verify defaults from settings are used
                assert service._stt_model == settings.SARVAM_STT_MODEL
                assert service._tts_model == settings.SARVAM_TTS_MODEL
                assert service._language == settings.SARVAM_LANGUAGE
                assert service._speaker == settings.SARVAM_SPEAKER
                assert service._tts_pace == settings.SARVAM_TTS_PACE
                assert service._tts_sample_rate == settings.SARVAM_TTS_SAMPLE_RATE

            asyncio.run(_test())

    def test_overrides_settings_with_kwargs(self):
        """Factory should allow overriding settings with kwargs."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service(
                    language="hi-IN",
                    speaker="custom-speaker"
                )
                assert isinstance(service, SarvamAiSpeechService)
                assert service._language == "hi-IN"
                assert service._speaker == "custom-speaker"

            asyncio.run(_test())

    def test_uses_openai_settings_defaults(self):
        """Factory should use settings for OpenAI defaults."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from src.flow_agent.configurations.config import settings

            import asyncio

            async def _test():
                service = await get_speech_service("openai")
                assert isinstance(service, OpenAiSpeechService)
                # Verify defaults from settings are used
                assert service._stt_model == settings.OPENAI_STT_MODEL
                assert service._tts_model == settings.OPENAI_TTS_MODEL
                assert service._language == settings.OPENAI_LANGUAGE
                assert service._speaker == settings.OPENAI_SPEAKER
                assert service._tts_pace == settings.OPENAI_TTS_PACE
                assert service._tts_sample_rate == settings.OPENAI_TTS_SAMPLE_RATE

            asyncio.run(_test())

    def test_overrides_openai_settings_with_kwargs(self):
        """Factory should allow overriding OpenAI settings with kwargs."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service(
                    "openai",
                    language="es-ES",
                    speaker="alloy"
                )
                assert isinstance(service, OpenAiSpeechService)
                assert service._language == "es-ES"
                assert service._speaker == "alloy"

            asyncio.run(_test())

    def test_uses_genai_settings_defaults(self):
        """Factory should use settings for GenAI defaults."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            from src.flow_agent.configurations.config import settings

            import asyncio

            async def _test():
                service = await get_speech_service("gemini")
                assert isinstance(service, GenAiSpeechService)
                # Verify defaults from settings are used
                assert service._stt_model == settings.GENAI_STT_MODEL
                assert service._tts_model == settings.GENAI_TTS_MODEL
                assert service._language == settings.GENAI_LANGUAGE
                assert service._speaker == settings.GENAI_SPEAKER
                assert service._tts_pace == settings.GENAI_TTS_PACE
                assert service._tts_sample_rate == settings.GENAI_TTS_SAMPLE_RATE

            asyncio.run(_test())

    def test_overrides_genai_settings_with_kwargs(self):
        """Factory should allow overriding GenAI settings with kwargs."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            import asyncio

            async def _test():
                service = await get_speech_service(
                    "gemini",
                    language="hi-IN",
                    speaker="Puck"
                )
                assert isinstance(service, GenAiSpeechService)
                assert service._language == "hi-IN"
                assert service._speaker == "Puck"

            asyncio.run(_test())


class TestGenAiSpeechService:
    """Test Google GenAI implementation (with mocked API calls)."""

    @pytest.fixture
    def mock_genai_client(self):
        """Create a mock GenAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def genai_service(self, mock_genai_client):
        """Create GenAiSpeechService with mocked client."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            service = GenAiSpeechService()
            service._client = mock_genai_client
            return service

    def test_init_requires_api_key(self):
        """Initialization should raise ValueError if API key is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GenAiSpeechService(api_key=None)

    def test_init_with_explicit_api_key(self):
        """Initialization should work with explicit API key."""
        service = GenAiSpeechService(api_key="explicit-key")
        assert service._api_key == "explicit-key"

    def test_init_with_default_params(self):
        """Initialization should use default parameters."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            service = GenAiSpeechService()
            assert service._stt_model == "gemini-3-flash-preview"
            assert service._tts_model == "gemini-2.5-flash-preview-tts"
            assert service._language == "en-IN"
            assert service._speaker == "Kore"
            assert service._tts_pace == 1.1
            assert service._tts_sample_rate == 24000

    def test_init_with_custom_params(self):
        """Initialization should accept custom parameters."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            service = GenAiSpeechService(
                stt_model="custom-stt",
                tts_model="custom-tts",
                language="hi-IN",
                speaker="custom-speaker",
                tts_pace=1.5,
                tts_sample_rate=44100,
            )
            assert service._stt_model == "custom-stt"
            assert service._tts_model == "custom-tts"
            assert service._language == "hi-IN"
            assert service._speaker == "custom-speaker"
            assert service._tts_pace == 1.5
            assert service._tts_sample_rate == 44100

    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, genai_service, mock_genai_client):
        """Test successful speech-to-text transcription."""
        mock_response = MagicMock()
        mock_response.text = "Transcribed text"
        mock_genai_client.models.generate_content.return_value = mock_response

        result = await genai_service.speech_to_text(b"fake audio")

        assert result == "Transcribed text"

        # Verify the API was called with correct parameters
        mock_genai_client.models.generate_content.assert_called_once()
        call_args = mock_genai_client.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-3-flash-preview"
        assert "Transcribe this audio clip" in call_args[1]["contents"]

    @pytest.mark.asyncio
    async def test_speech_to_text_with_mp3_extension(
        self, genai_service, mock_genai_client
    ):
        """Test speech-to-text with MP3 file extension."""
        mock_response = MagicMock()
        mock_response.text = "Transcribed text"
        mock_genai_client.models.generate_content.return_value = mock_response

        result = await genai_service.speech_to_text(b"fake audio", file_extension=".mp3")

        assert result == "Transcribed text"

        # Verify the API was called (we just check it was called, not the internal structure)
        mock_genai_client.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_to_text_failure(self, genai_service, mock_genai_client):
        """Test speech-to-text when API call fails."""
        # Mock exception
        mock_genai_client.models.generate_content.side_effect = Exception("API Error")

        # Call the method
        result = await genai_service.speech_to_text(b"fake audio data")

        # Verify None is returned on failure
        assert result is None

    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, genai_service, mock_genai_client):
        """Test successful text-to-speech conversion."""
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"fake pcm audio"
        mock_part.inline_data = mock_inline_data
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_genai_client.models.generate_content.return_value = mock_response

        result = await genai_service.text_to_speech("Hello world")

        assert result is not None
        assert len(result) == 1
        import base64
        decoded = base64.b64decode(result[0])
        assert decoded == b"fake pcm audio"

        # Verify the API was called with correct parameters
        mock_genai_client.models.generate_content.assert_called_once()
        call_args = mock_genai_client.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-flash-preview-tts"
        assert call_args[1]["contents"] == "Hello world"

    @pytest.mark.asyncio
    async def test_text_to_speech_failure(self, genai_service, mock_genai_client):
        """Test text-to-speech when API call fails."""
        # Mock exception
        mock_genai_client.models.generate_content.side_effect = Exception("API Error")

        # Call the method
        result = await genai_service.text_to_speech("Hello world")

        # Verify None is returned on failure
        assert result is None

    def test_provider_name(self, genai_service):
        """Test provider_name property."""
        assert genai_service.provider_name == "gemini"

