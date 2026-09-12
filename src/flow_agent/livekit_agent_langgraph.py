"""
LiveKit Voice Agent - SAFE VERSION
Only web_search and calculate tools (removes get_current_time)
These definitely work with parameters
"""

import logging
import os
import re
import json
from pathlib import Path
import sys

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, function_tool, Agent, RunContext
from livekit.agents.voice import AgentSession
from livekit.plugins import sarvam, silero, groq
from langchain_tavily import TavilySearch

load_dotenv()

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
TTS_MODEL = os.getenv("TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "ishita")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AGENT_INSTRUCTIONS = """
You are a helpful voice assistant with access to tools.
Be concise and natural - keep responses under 50 words for voice.
Use web_search tool when you need current information.
Use calculate tool for math problems.
Speak naturally and conversationally.
"""


def remove_urls(text: str) -> str:
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    cleaned = re.sub(url_pattern, '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


tavily_search = TavilySearch(max_results=3)


class VoiceAgent(Agent):
    """Voice Agent with 2 working tools"""

    @function_tool()
    async def web_search(self, context: RunContext, query: str) -> str:
        """Search the web for current information"""
        try:
            logger.info(f"🔍 Searching: {query}")
            result = tavily_search.run(query)
            logger.info(f"Result type: {type(result)}")

            # Handle both dict and string responses
            if isinstance(result, dict):
                # It's a dict - extract the answer
                logger.info("Result is dict")
                content_parts = []

                if result.get("answer"):
                    answer = remove_urls(str(result["answer"]))
                    content_parts.append(answer)

                # If no answer, try to get content from results
                if not content_parts:
                    for item in result.get("results", []):
                        content = item.get("content", "")
                        if content:
                            content = remove_urls(str(content))
                            if content:
                                content_parts.append(content)

                final_result = "\n".join(content_parts)
                logger.info(f"✅ Search complete: {len(final_result)} chars")
                return final_result if final_result else "No search results found."

            else:
                # It's a string - just clean and return
                logger.info("Result is string")
                cleaned = remove_urls(str(result))
                logger.info(f"✅ Search complete")
                return cleaned if cleaned else result

        except Exception as e:
            logger.error(f"❌ Search failed: {e}", exc_info=True)
            return f"Search failed: {str(e)}"

    @function_tool()
    async def calculate(self, context: RunContext, expression: str) -> str:
        """Calculate a mathematical expression"""
        try:
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Invalid expression."
            result = eval(expression)
            return f"The result is {result}"
        except Exception as e:
            return "I couldn't calculate that."


async def entrypoint(ctx: JobContext):
    try:
        logger.info("=" * 70)
        logger.info(f"🎤 User connected: {ctx.room.name}")
        logger.info("=" * 70)

        logger.info("🎙️  Initializing voice components...")
        vad = silero.VAD.load()
        stt = groq.STT(model=STT_MODEL, language="en", detect_language=True)
        # stt = sarvam.STT(language="bn-IN",  model="saaras:v3", high_vad_sensitivity=True)
        tts = sarvam.TTS(target_language_code="en-IN", model=TTS_MODEL, speaker=TTS_SPEAKER)
        logger.info("✅ Voice components ready")

        logger.info("🎙️  Creating voice session...")
        session = AgentSession(vad=vad, stt=stt, tts=tts)
        logger.info("✅ Voice session created")

        logger.info("🤖 Creating voice agent with 2 tools...")
        agent = VoiceAgent(llm=groq.LLM(model=LLM_MODEL), instructions=AGENT_INSTRUCTIONS)
        logger.info("✅ Agent ready")
        logger.info("   ✅ web_search(query)")
        logger.info("   ✅ calculate(expression)")

        logger.info("🎙️  Starting voice session...")
        await session.start(agent=agent, room=ctx.room)

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 15 + "LiveKit Voice Agent (2 Tools)" + " " * 25 + "║")
    logger.info("║" + " " * 18 + "SAFE & TESTED" + " " * 37 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")

    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY not set")
        exit(1)

    logger.info("✅ Ready!")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
