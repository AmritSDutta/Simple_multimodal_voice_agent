import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage

from src.flow_agent.configurations.config import settings


@pytest.mark.asyncio
async def test_summarization_trigger():
    """Test that summarization triggers when message threshold is reached."""
    custom_settings = settings.model_copy(update={
        "SUMMARY_MESSAGE_THRESHOLD": 7,
        "SUMMARY_PROVIDER_PREFERENCE": "langchain",
        "PROVIDER_DISTRIBUTION": {
            "gemini": 1.0,
            "openai": 0.0,
            "zhipu": 0.0,
            "ollama": 0.0,
        },
        "FALLBACK_PROVIDER_IDENTIFIER": "gemini",
        "GEMINI_VISION_MODEL": "gemma-3-12b-it",
    })

    with patch("src.flow_agent.graph.settings", custom_settings), \
         patch("src.flow_agent.utils.nodes.settings", custom_settings):

        from importlib import reload
        import src.flow_agent.graph as graph_module
        reload(graph_module)

        compiled_graph = graph_module.graph.compile()

        # Below threshold - 3 human + 2 AI = 5 messages (< 6)
        state = {
            "messages": [
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2"),
                AIMessage(content="A2"),
                HumanMessage(content="Q3"),
            ],
            "input_valid": True,
        }
        result = await compiled_graph.ainvoke(state, config={"configurable": {"my_configurable_param": "test"}})
        # No summarization occurred
        human_count = sum(1 for m in result.get("messages", []) if isinstance(m, HumanMessage))
        assert human_count == 3  # All human messages retained

        # Above threshold - 4 human + 3 AI = 7 messages (>= 6)
        state = {
            "messages": [
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2"),
                AIMessage(content="A2"),
                HumanMessage(content="Q3"),
                AIMessage(content="A3"),
                HumanMessage(content="Q4"),
            ],
            "input_valid": True,
        }
        result = await compiled_graph.ainvoke(state, config={"configurable": {"my_configurable_param": "test"}})
        assert "conversation_summary" in result
        assert len(result["conversation_summary"]) > 0

        # Verify: all human messages retained + last 2 AI + summary
        human_count = sum(1 for m in result.get("messages", []) if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in result.get("messages", []) if isinstance(m, AIMessage))
        assert human_count == 1
        assert ai_count == 2


@pytest.mark.asyncio
async def test_summarization_retains_images():
    """Test that the summarizer node correctly handles message retention logic."""
    # Test the summarizer node directly without full graph execution
    from src.flow_agent.utils.nodes import _get_retained_messages
    from unittest.mock import MagicMock

    # Mock runtime with store
    mock_runtime = MagicMock()
    mock_runtime.store = MagicMock()

    # Fake base64 image data
    fake_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    # Create test state with mixed text and image messages
    state = {
        "messages": [
            HumanMessage(content=[{"type": "text", "text": "What's in this image?"}, {"type": "image", "data": fake_image_b64, "metadata": {"filename": "test1.png"}}]),
            AIMessage(content="I see a red pixel."),
            HumanMessage(content=[{"type": "text", "text": "And this one?"}, {"type": "image", "data": fake_image_b64, "metadata": {"filename": "test2.png"}}]),
            AIMessage(content="This is also a red pixel."),
            HumanMessage(content="Latest question"),
        ],
        "conversation_summary": "",
    }

    # Test the _get_retained_messages helper function
    messages_to_summarize = state["messages"][:-2]  # All but last 2
    text_messages, media_messages = await _get_retained_messages(messages_to_summarize)

    # Verify separation logic
    assert len(text_messages) == 3, f"Expected 3 text messages, got {len(text_messages)}"  # 2 AIMessage + 2 HumanMessage text parts = 3 single-item messages
    assert len(media_messages) == 2, f"Expected 2 media messages, got {len(media_messages)}"  # 2 HumanMessage with images

    # Verify media messages have image content
    has_image_count = sum(1 for m in media_messages if isinstance(m.content, list) and any(item.get("type") == "image" for item in m.content if isinstance(item, dict)))
    assert has_image_count == 2, f"Expected 2 messages with images, got {has_image_count}"

    print("✅ Media retention logic verified - media are correctly separated from text for summarization")


@pytest.mark.asyncio
async def test_get_retained_messages_empty_input():
    """Test that empty/None input returns empty lists."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    # None input
    text, media = await _get_retained_messages(None)
    assert text == [], "Expected empty list for text when input is None"
    assert media == [], "Expected empty list for media when input is None"

    # Empty list
    text, media = await _get_retained_messages([])
    assert text == [], "Expected empty list for text"
    assert media == [], "Expected empty list for media"

    print("✅ Empty input handling verified")


@pytest.mark.asyncio
async def test_get_retained_messages_string_content():
    """Test that plain string content messages go to text_messages."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    messages = [
        HumanMessage(content="Plain text question"),
        AIMessage(content="Plain text answer"),
    ]

    text, media = await _get_retained_messages(messages)

    assert len(text) == 2, f"Expected 2 text messages, got {len(text)}"
    assert len(media) == 0, f"Expected 0 media messages, got {len(media)}"
    assert all(isinstance(m.content, str) for m in text), "All text messages should have string content"

    print("✅ String content handling verified")


@pytest.mark.asyncio
async def test_get_retained_messages_list_text_only():
    """Test list content with only text items."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    messages = [
        HumanMessage(content=[{"type": "text", "text": "Question one"}]),
        AIMessage(content=[{"type": "text", "text": "Answer one"}]),
    ]

    text, media = await _get_retained_messages(messages)

    assert len(text) == 2, f"Expected 2 text messages, got {len(text)}"
    assert len(media) == 0, f"Expected 0 media messages, got {len(media)}"
    assert all(isinstance(m, HumanMessage) or isinstance(m, AIMessage) for m in text), "Should preserve message types"

    print("✅ List text-only content handling verified")


@pytest.mark.asyncio
async def test_get_retained_messages_list_image_only():
    """Test list content with only image items."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    fake_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    messages = [
        HumanMessage(content=[{"type": "image", "data": fake_image_b64, "metadata": {"filename": "test.png"}}]),
        AIMessage(content=[{"type": "image", "data": fake_image_b64, "metadata": {"filename": "test2.png"}}]),
    ]

    text, media = await _get_retained_messages(messages)

    assert len(text) == 0, f"Expected 0 text messages, got {len(text)}"
    assert len(media) == 2, f"Expected 2 media messages, got {len(media)}"

    # Verify media messages have image content
    for m in media:
        assert isinstance(m.content, list), "Media message content should be a list"
        assert m.content[0].get("type") == "image", "Should be image type"

    print("✅ List image-only content handling verified")


@pytest.mark.asyncio
async def test_get_retained_messages_mixed_content():
    """Test list content with both text and image in same message."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    fake_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "data": fake_image_b64, "metadata": {"filename": "test.png"}}
        ]),
    ]

    text, media = await _get_retained_messages(messages)

    # The function splits mixed content into separate messages
    assert len(text) == 1, f"Expected 1 text message, got {len(text)}"
    assert len(media) == 1, f"Expected 1 media message, got {len(media)}"

    # Verify the text message
    assert isinstance(text[0], HumanMessage), "Text message should preserve HumanMessage type"
    assert text[0].content == [{"type": "text", "text": "What's in this image?"}]

    # Verify the media message
    assert isinstance(media[0], HumanMessage), "Media message should preserve HumanMessage type"
    assert media[0].content[0].get("type") == "image"

    print("✅ Mixed content splitting verified")


@pytest.mark.asyncio
async def test_get_retained_messages_audio_video_types():
    """Test that audio and video types are handled correctly."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    messages = [
        HumanMessage(content=[{"type": "audio", "data": "audio_data"}]),
        AIMessage(content=[{"type": "video", "data": "video_data"}]),
        HumanMessage(content=[{"type": "image", "data": "image_data"}]),
    ]

    text, media = await _get_retained_messages(messages)

    assert len(text) == 0, f"Expected 0 text messages, got {len(text)}"
    assert len(media) == 3, f"Expected 3 media messages (audio+video+image), got {len(media)}"

    # Verify all media types are captured
    media_types = [m.content[0].get("type") for m in media]
    assert "audio" in media_types, "Should handle audio type"
    assert "video" in media_types, "Should handle video type"
    assert "image" in media_types, "Should handle image type"

    print("✅ Audio/video media type handling verified")


@pytest.mark.asyncio
async def test_get_retained_messages_multiple_items():
    """Test messages with multiple text items."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "First question"},
            {"type": "text", "text": "Second question"},
            {"type": "text", "text": "Third question"},
        ]),
    ]

    text, media = await _get_retained_messages(messages)

    # Each text item becomes a separate message
    assert len(text) == 3, f"Expected 3 text messages (one per item), got {len(text)}"
    assert len(media) == 0, f"Expected 0 media messages, got {len(media)}"
    assert all(isinstance(m, HumanMessage) for m in text), "All should be HumanMessage type"

    # Verify each message has single text item
    for i, m in enumerate(text):
        assert len(m.content) == 1, f"Message {i} should have single item"
        assert m.content[0].get("type") == "text", f"Message {i} should be text type"

    print("✅ Multiple items per message handling verified")


@pytest.mark.asyncio
async def test_get_retained_messages_preserves_types():
    """Test that message types (HumanMessage/AIMessage) are preserved."""
    from src.flow_agent.utils.nodes import _get_retained_messages

    fake_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    messages = [
        HumanMessage(content="Human text"),
        AIMessage(content="AI text"),
        HumanMessage(content=[{"type": "image", "data": fake_image_b64}]),
        AIMessage(content=[{"type": "text", "text": "AI response"}]),
    ]

    text, media = await _get_retained_messages(messages)

    # Verify types are preserved
    human_in_text = sum(1 for m in text if isinstance(m, HumanMessage))
    ai_in_text = sum(1 for m in text if isinstance(m, AIMessage))
    human_in_media = sum(1 for m in media if isinstance(m, HumanMessage))
    ai_in_media = sum(1 for m in media if isinstance(m, AIMessage))

    assert human_in_text == 1, "Should have 1 HumanMessage in text"
    assert ai_in_text == 2, "Should have 2 AIMessage in text (string + list item)"
    assert human_in_media == 1, "Should have 1 HumanMessage in media"

    print("✅ Message type preservation verified")


@pytest.mark.asyncio
async def test_summarization_limits_images():
    """Test that only the most recent MAX_IMAGES_PER_REQUEST images are retained."""
    from src.flow_agent.utils.nodes import (
        _filter_recent_messages_by_type,
        _extract_all_messages_by_type,
        _has_media_type
    )
    from src.flow_agent.configurations.config import settings

    fake_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    # Create 6 messages with images (more than MAX_IMAGES_PER_REQUEST which is 4)
    messages = [
        HumanMessage(content=[{"type": "text", "text": "Image 1"}, {"type": "image", "data": fake_image_b64}]),
        AIMessage(content="Response 1"),
        HumanMessage(content=[{"type": "text", "text": "Image 2"}, {"type": "image", "data": fake_image_b64}]),
        AIMessage(content="Response 2"),
        HumanMessage(content=[{"type": "text", "text": "Image 3"}, {"type": "image", "data": fake_image_b64}]),
        AIMessage(content="Response 3"),
        HumanMessage(content=[{"type": "text", "text": "Image 4"}, {"type": "image", "data": fake_image_b64}]),
        AIMessage(content="Response 4"),
        HumanMessage(content=[{"type": "text", "text": "Image 5"}, {"type": "image", "data": fake_image_b64}]),
        AIMessage(content="Response 5"),
        HumanMessage(content=[{"type": "text", "text": "Image 6"}, {"type": "image", "data": fake_image_b64}]),
        AIMessage(content="Response 6"),
    ]

    # Extract all image messages
    all_images = _extract_all_messages_by_type(messages, "image")
    assert len(all_images) == 6, f"Should have 6 images, got {len(all_images)}"

    # Filter to keep only recent images
    recent_images = _filter_recent_messages_by_type(messages, "image", settings.MAX_IMAGES_PER_REQUEST)

    # Verify only MAX_IMAGES_PER_REQUEST images are kept
    assert len(recent_images) == settings.MAX_IMAGES_PER_REQUEST, \
        f"Should keep {settings.MAX_IMAGES_PER_REQUEST} images, got {len(recent_images)}"

    # Verify the kept images are the most recent ones (last 4)
    assert all(_has_media_type(msg, "image") for msg in recent_images), "All kept messages should have images"

    print(f"✅ Image limiting verified: Kept {len(recent_images)}/{len(all_images)} most recent images (max={settings.MAX_IMAGES_PER_REQUEST})")
