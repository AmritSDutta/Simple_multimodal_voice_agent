import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam, google, groq, langchain, silero

# Load environment variables
load_dotenv()
# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # Agent prompt
            instructions="""
                You are a helpful voice assistant.
                Be friendly, concise, and conversational. Speak 
                naturally as if you're having a real conversation.
            """,
            # voice activity detector
            vad=silero.VAD.load(),
            # STT
            stt=groq.STT(
                model="whisper-large-v3-turbo",
                language="en"
            ),

            # LLM
            llm=groq.LLM(model="llama-3.1-8b-instant"),

            # TTS
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="ishita"),
        )

    async def on_enter(self):
        """Called when user joins - agent starts the conversation"""
        await self.session.generate_reply()


async def entrypoint(ctx: JobContext):
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
