import pytest
from arize.experiments.evaluators.types import Score
from langchain_core.messages import HumanMessage
from phoenix.evals import ClassificationEvaluator
from phoenix.evals.llm import LLM

from src.flow_agent.configurations.config import settings
from src.flow_agent.graph import graph

judge_llm = LLM(provider="openai", model="gpt-5-nano")

vision_eval = ClassificationEvaluator(
    name="vision_accuracy",
    llm=judge_llm,
    prompt_template="""
        You are judging a vision assistant's answer to a question about an OLAP/data cube image.
        
        If the answer correctly describes data cubes, OLAP concepts, or multidimensional data visualization, reply "correct".
        If it is factually wrong, completely unrelated, or hallucinates non-existent elements, reply "incorrect".
        
        User prompt: {input}
        Model answer: {output}
    """,
    choices={
        "incorrect": 0.0,
        "correct": 1.0
    },
)


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

    image_base64 = image_to_base64_fixture(str(resources_path / "olap.png"))
    prompt = "Explain the attached image in 10 words"

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "data": image_base64,
            "metadata": {"filename": "olap.png"},
            "source_type": "base64",
            "mime_type": "image/png",
        },
    ]
    compiled_graph = graph.compile()
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

    # ---- Extract model answer (adjust if your state shape differs) ----
    answer = result["messages"][-1].content

    # ---- Phoenix LLM-as-Judge ----
    scores: list[Score] = vision_eval.evaluate(
        {"input": prompt, "output": answer}
    )
    score = scores[0].score
    label = scores[0].label

    # ---- Gate ----
    threshold = 0.9  # require "correct"
    assert score >= threshold, f"Score {score} < {threshold}\nAnswer: {answer}"
    assert label.lower() == "correct"
