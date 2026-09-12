import logging
import os

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam, google, groq, langchain, silero
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()
# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

ollama_client = AsyncOpenAI(
    api_key=os.getenv("SARVAM_API_KEY"),
    base_url="https://api.sarvam.ai/v1"
)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # Your agent's personality and instructions
            instructions="""
                You are a helpful voice assistant.
                Be friendly, concise, and conversational.
                Speak naturally as if you're having a real conversation.
            """,
            vad=silero.VAD.load(),
            stt=groq.STT(
                model="whisper-large-v3-turbo",
                language="en",
            ),

            llm=openai.LLM.with_ollama(
                model='sarvam-m',
                base_url="https://api.sarvam.ai",
                client=AsyncOpenAI(
                    api_key=os.getenv("SARVAM_API_KEY"),
                    base_url="https://api.sarvam.ai/v1"
                )
            ),

            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="aditya"
            ),
        )

    async def on_enter(self):
        """Called when user joins - agent starts the conversation"""
        await self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """Main entry point - LiveKit calls this when a user connects"""
    logger.info(f"User connected to room: {ctx.room.name}")

    # Create and start the agent session
    session = AgentSession()
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room
    )


if __name__ == "__main__":
    # Run the agent
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
