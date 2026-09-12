"""
LiveKit Voice Agent with Context Optimization
Demonstrates context compaction for long-running specialized tasks.

This agent implements three context management strategies:
1. Interruption-based truncation (automatic)
2. Periodic summarization (proactive)
3. Message filtering (noise reduction)

Usage:
    python src/livekit_context_optimized.py

Environment Variables:
    GROQ_API_KEY: Groq API key for LLM/STT
    SARVAM_API_KEY: SarvamAI API key for TTS
    TAVILY_API_KEY: Tavily API key for web search (required for web_search tool)
    CONTEXT_SUMMARIZE_THRESHOLD_ITEMS: Trigger summarization (default: 20)
    CONTEXT_KEEP_LAST_N_TURNS: Uncompressed turns to keep (default: 10)
    CONTEXT_SUMMARIZE_EVERY_N_TURNS: Proactive check frequency (default: 15)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool, RunContext,
)
from livekit.plugins import groq, sarvam, silero, google, openai
from openai import AsyncClient, AsyncOpenAI
from langchain_tavily import TavilySearch
import re

# Import context management modules
from src.livekit_context_manager import ContextManager
from src.specialized_domain_state import SpecializedDomainState

load_dotenv()

logger = logging.getLogger("voice-agent-optimized")
logger.setLevel(logging.INFO)

ollama_client = AsyncOpenAI(
    api_key=os.getenv("OLLAMA_API_KEY"),
    base_url="https://ollama.com/v1"
)
'''
           instructions="""
                                      You are a specialized voice assistant teacher for english grammar viva tasks.
                        Be friendly, concise, and conversational to 5th grade student.
                        You must be well versed with author -> J. C. Nesfield's grammar book part 3, specifically Nouns,
                        Gerund, past participle, gender, plural forms etc.
                        Generate 10 high quality concise question and ask the student one by one, 
                        once the student finished , provide expected answers  in concise way.
                        Dont repeat similar questions.
                        If they reference previous information, check their preferences first.
                        Your answers should be always succinct and to the point.
           """,
'''
'''
llm=openai.LLM.with_ollama(
    model='nemotron-3-nano:30b-cloud',
    base_url="https://ollama.com",
    client=ollama_client
),
'''


def remove_urls(text: str) -> str:
    """Remove URLs from text to clean up search results"""
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    cleaned = re.sub(url_pattern, '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


# Initialize Tavily search client
tavily_search = TavilySearch(max_results=3)


@dataclass
class AgentUserData:
    """Session userdata for context-optimized agent"""
    ctx: Optional[JobContext] = None
    domain_state: Optional[SpecializedDomainState] = None


class OptimizedTaskAgent(Agent):
    """
    Voice agent with proactive context management.

    Features:
    - Automatic context filtering (removes system noise)
    - Periodic summarization (prevents token limits)
    - External state storage (reduces context bloat)
    - Interruption handling (automatic truncation)
    """

    def __init__(self):
        super().__init__(

            instructions="""
            
                        You are a specialized voice assistant for technical interview tasks.
               Be friendly, concise, and conversational.
               You must be well versed with Dotnet, datastructure, algorithm.
               You maintain context efficiently for long conversations.
               Generate high quality concise question and answer, and evaluate, give rating.
               Always provide expected answers clue in concise way.
               When users ask for calculations or web searches, use the available tools.
               If they reference previous information, check their preferences first.
               Your answers should be always succinct and to the point.
                    """,
            # STT - Speech to Text (Groq Whisper)
            vad=silero.VAD.load(),
            stt=groq.STT(
                model="whisper-large-v3-turbo",
                language="en",
            ),

            # LLM - The "brain" (Groq Llama)

            llm=google.LLM(
                model='gemini-2.5-flash',
                api_key=os.getenv('GEMINI_API_KEY')
                #base_url="https://ollama.com",
                #client=ollama_client
            ),
            # TTS - Text to Speech (SarvamAI)
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="aditya"
            ),

            # Interruptions enabled by default for automatic truncation
            allow_interruptions=True,
        )

        # Context management settings
        self._turn_counter = 0
        self._SUMMARIZE_EVERY_N_TURNS = int(
            os.getenv("CONTEXT_SUMMARIZE_EVERY_N_TURNS", "5")
        )
        self._SUMMARIZE_THRESHOLD_ITEMS = int(
            os.getenv("CONTEXT_SUMMARIZE_THRESHOLD_ITEMS", "10")
        )
        self._KEEP_LAST_N_TURNS = int(
            os.getenv("CONTEXT_KEEP_LAST_N_TURNS", "3")
        )

        logger.info(f"📊 Context settings:")
        logger.info(f"  - Summarize every N turns: {self._SUMMARIZE_EVERY_N_TURNS}")
        logger.info(f"  - Threshold items: {self._SUMMARIZE_THRESHOLD_ITEMS}")
        logger.info(f"  - Keep last N turns: {self._KEEP_LAST_N_TURNS}")

    async def on_enter(self):
        """
        Called when user joins - initialize context management.

        Checks if summarization is needed and generates initial response.
        """
        logger.info("👤 User joined session")

        # Domain state is already initialized in entrypoint
        # Just verify it exists
        if self.session.userdata.domain_state is None:
            self.session.userdata.domain_state = SpecializedDomainState()
            logger.info("✅ Initialized specialized domain state")
        else:
            logger.info("✅ Domain state already initialized")

        # Check if summarization needed (proactive check on entry)
        await ContextManager.summarize_if_needed(
            agent=self,
            threshold_items=self._SUMMARIZE_THRESHOLD_ITEMS,
            keep_last_turns=3,
        )

        # Log context stats
        stats = ContextManager.get_context_stats(self)
        logger.info(f"📊 Context stats: {stats['total_items']} items")

        await self.session.generate_reply()

    async def on_user_turn_ended(self):
        """
        Called after each user turn - periodic context check.

        Every N turns, checks if summarization is needed to prevent
        context window overflow.
        """
        self._turn_counter += 1

        # Every N turns, check if summarization needed
        if self._turn_counter % self._SUMMARIZE_EVERY_N_TURNS == 0:
            logger.info(f"🔄 Turn {self._turn_counter}: Checking context...")

            stats = ContextManager.get_context_stats(self)
            logger.info(f"📊 Context stats: {stats['total_items']} items")

            await ContextManager.summarize_if_needed(
                agent=self,
                threshold_items=self._SUMMARIZE_THRESHOLD_ITEMS,
                keep_last_turns=3,
            )

    @function_tool()
    async def web_search(self, context: RunContext, query: str) -> str:
        """
        Search the web for current information.

        Args:
            query: Search query

        Returns:
            Search results summary
        """
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
        """
        Perform mathematical calculations.

        Args:
            expression: Mathematical expression (e.g., "2 + 2")

        Returns:
            Calculation result
        """
        # Compact context before tool execution
        compacted_ctx = await ContextManager.get_compacted_context(
            agent=self,
            keep_last_n_turns=self._KEEP_LAST_N_TURNS,
            exclude_system=True
        )

        # Update agent's context to compacted version
        self._chat_ctx = compacted_ctx

        try:
            # Safe evaluation of mathematical expressions
            result = eval(expression, {"__builtins__": {}}, {})
            logger.info(f"🧮 Calculated: {expression} = {result}")
            return f"Result: {result}"
        except Exception as e:
            logger.error(f"❌ Calculation error: {e}")
            return f"Error: {str(e)}"

    @function_tool()
    async def save_preference(
            self,
            context: RunContext[AgentUserData],
            key: str,
            value: str
    ) -> str:
        """
        Save user preference to external state (not chat context).

        This stores preferences outside the conversation to avoid
        consuming context tokens.

        Args:
            key: Preference key (e.g., "language", "units")
            value: Preference value

        Returns:
            Confirmation message
        """
        context.userdata.domain_state.set_user_preference(key, value)
        logger.info(f"💾 Saved preference: {key} = {value}")
        return f"Saved: {key} = {value}"

    @function_tool()
    async def get_preference(
            self,
            context: RunContext[AgentUserData],
            key: str
    ) -> str:
        """
        Retrieve preference from external state.

        Args:
            key: Preference key to retrieve

        Returns:
            Preference value or "not set" message
        """
        value = context.userdata.domain_state.get_user_preference(key)
        if value:
            logger.info(f"📖 Retrieved preference: {key} = {value}")
            return value
        else:
            logger.info(f"⚠️ Preference not found: {key}")
            return f"Preference '{key}' is not set"

    @function_tool()
    async def get_context_stats(self, placeholder: str = "") -> str:
        """
        Get current context statistics (for debugging).

        Args:
            placeholder: Placeholder parameter (no input required, empty string by default)

        Returns:
            Human-readable context statistics
        """
        stats = ContextManager.get_context_stats(self)

        domain_state = self.session.userdata.domain_state
        state_summary = domain_state.to_summary()

        return (
            f"Context Statistics:\n"
            f"- Total items: {stats['total_items']}\n"
            f"- Type breakdown: {stats['type_breakdown']}\n"
            f"- Domain state: {state_summary}\n"
            f"- Session duration: {domain_state.get_session_duration():.1f}s"
        )


async def entrypoint(ctx: JobContext):
    """
    Main entry point - LiveKit calls this when a user connects.

    Args:
        ctx: Job context with room connection
    """
    logger.info(f"🔌 User connected to room: {ctx.room.name}")

    # Initialize userdata with domain state
    userdata = AgentUserData(
        ctx=ctx,
        domain_state=SpecializedDomainState()
    )

    # Create and start the agent session with userdata
    session = AgentSession[AgentUserData](userdata=userdata)
    await session.start(
        agent=OptimizedTaskAgent(),
        room=ctx.room
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Print startup banner
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 15 + "LiveKit Context-Optimized Agent" + " " * 22 + "║")
    logger.info("║" + " " * 20 + "Long-Running Support" + " " * 32 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    logger.info("✅ Features enabled:")
    logger.info("  - Automatic context filtering")
    logger.info("  - Periodic summarization")
    logger.info("  - External state storage")
    logger.info("  - Interruption-based truncation")
    logger.info("")

    # Start the agent
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
