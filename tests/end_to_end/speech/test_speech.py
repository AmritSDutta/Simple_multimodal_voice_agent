import pytest
import base64
from src.flow_agent.speech import get_speech_service, SpeechService
from difflib import SequenceMatcher


@pytest.mark.asyncio
async def test_sarvam_speech_e2e():
    test_input: str = "hello sarvam, how are you?"
    sarvam_speech:  SpeechService = await get_speech_service('sarvam')
    audio: list[str] = await sarvam_speech.text_to_speech(test_input)
    assert audio is not None

    audio_bytes = base64.b64decode(audio[0])
    text: str = await sarvam_speech.speech_to_text(audio_bytes)
    assert text is not None
    similarity = SequenceMatcher(None, test_input.lower(), text.lower()).ratio()
    assert similarity >= 0.8, f"Text similarity {similarity:.2%} below 80% threshold"


@pytest.mark.asyncio
async def test_genai_speech_e2e():
    test_input: str = "hello gemini, how are you?"
    gemini_speech:  SpeechService = await get_speech_service('gemini')
    audio: list[str] = await gemini_speech.text_to_speech(test_input)
    assert audio is not None

    audio_bytes = base64.b64decode(audio[0])
    text: str = await gemini_speech.speech_to_text(audio_bytes)
    assert text is not None
    similarity = SequenceMatcher(None, test_input.lower(), text.lower()).ratio()
    assert similarity >= 0.8, f"Text similarity {similarity:.2%} below 80% threshold"


@pytest.mark.asyncio
async def test_openai_speech_e2e():
    test_input: str = "hello openai, how are you?"
    openai_speech:  SpeechService = await get_speech_service('openai')
    audio: list[str] = await openai_speech.text_to_speech(test_input)
    assert audio is not None

    audio_bytes = base64.b64decode(audio[0])
    text: str = await openai_speech.speech_to_text(audio_bytes)
    assert text is not None
    similarity = SequenceMatcher(None, test_input.lower(), text.lower()).ratio()
    assert similarity >= 0.8, f"Text similarity {similarity:.2%} below 80% threshold"
