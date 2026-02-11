from unittest.mock import patch

import pytest

from langchain_core.messages import HumanMessage
from src.flow_agent.configurations.config import settings
from src.flow_agent.graph import graph


@pytest.mark.asyncio
async def test_graph_flow_text_only():
    compiled_graph = graph.compile()
    result = await compiled_graph.ainvoke(
        {
            # initial State fields go here
            "messages": [
                HumanMessage(content='why sku is blue in 10 words'),
            ]
        },
        config={
            "configurable": {
                "my_configurable_param": "test-value",
            }
        },
    )

    assert result is not None


@pytest.mark.asyncio
async def test_graph_flow_with_image(resources_path, image_to_base64_fixture):
    custom_settings = settings.model_copy(update={
        "MAX_TRY": 1,
        "SLEEP": 0,
        "PROVIDER_DISTRIBUTION": {
            "gemini": 1.0,
            "openai": 0.0,
            "zhipu": 0.0,
            "ollama": 0.0,
        },
        "FALLBACK_PROVIDER_IDENTIFIER": "gemini",
        "GEMINI_VISION_MODEL": "gemma-3-12b-it",
    })
    compiled_graph = graph.compile()
    image_base64 = image_to_base64_fixture(str(resources_path / 'olap.png'))
    content = [{"type": "text", "text": 'explain the attached image  in 10 words'}]
    mime_type = "image/png"
    b64_data = image_base64
    content.append(
        {
            "type": "image",
            "data": b64_data,
            "metadata": {"filename": 'olap.png'},
            "source_type": "base64",
            "mime_type": mime_type,
        }
    )

    result = await compiled_graph.ainvoke(
        {
            # initial State fields go here
            "messages": [
                HumanMessage(content=content),
            ]
        },
        config={
            "configurable": {
                "my_configurable_param": "test-value",
            }
        },
    )

    assert result is not None


@pytest.mark.asyncio
async def test_graph_flow_with_genai_node(resources_path, image_to_base64_fixture):
    """
    Test using the native Google GenAI reasoning node instead of LangChain.
    Uses mocked settings to override REASONING_NODE_PREFERENCE.
    """
    # Copy existing settings and only override what we need
    custom_settings = settings.model_copy(update={
        "MAX_TRY": 1,
        "SLEEP": 0,
        "REASONING_NODE_PREFERENCE": "genai",
    })

    # Patch settings in both the graph module and nodes module
    with patch("src.flow_agent.graph.settings", custom_settings), \
         patch("src.flow_agent.utils.nodes.settings", custom_settings):

        # Import graph fresh to use mocked settings
        from importlib import reload
        import src.flow_agent.graph as graph_module
        reload(graph_module)

        compiled_graph = graph_module.graph.compile()
        image_base64 = image_to_base64_fixture(str(resources_path / 'olap.png'))
        content = [{"type": "text", "text": 'explain the attached image  in 10 words'}]
        mime_type = "image/png"
        b64_data = image_base64
        content.append(
            {
                "type": "image",
                "data": b64_data,
                "metadata": {"filename": 'olap.png'},
                "source_type": "base64",
                "mime_type": mime_type,
            }
        )
        result = await compiled_graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=content),
                ]
            },
            config={
                "configurable": {
                    "my_configurable_param": "test-value",
                }
            },
        )

        assert result is not None
        assert "messages" in result
