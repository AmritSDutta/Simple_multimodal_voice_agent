"""
Simplest Screen Sharing LiveKit App
Utilizes existing components and patterns from livekit_agent.py.
Monitors for SCREEN_SHARE tracks and prepares for vision processing.

Required Environment Variables:
- OPENAI_API_KEY: Required for GPT-4o vision capabilities.
- SARVAM_API_KEY: Required for TTS.
- GROQ_API_KEY: Required for STT.
"""

import logging
import asyncio
import os
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam, silero, groq, google
from livekit.rtc import TrackSource, VideoStream, TrackKind

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("screen-share-agent")
logger.setLevel(logging.INFO)


class ScreenAwareVoiceAgent(Agent):
    """
    A simple voice agent that uses existing components.
    It's configured with GPT-4o which is vision-capable.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful voice assistant that can see the user's shared screen.
                When the user shares their screen, you will be able to analyze its content.
                Be friendly, concise, and conversational.
                Always acknowledge when you see a screen being shared.
                If the user asks 'What do you see?', describe the screen share content.
            """,
            vad=silero.VAD.load(),
            stt=groq.STT(
                model="whisper-large-v3-turbo",
                language="en"
            ),
            # Use GPT-4o for vision capabilities
            llm=google.LLM(model="gemini-2.5-flash-lite", api_key=os.getenv('GEMINI_API_KEY')),

            # Using Sarvam for high-quality Indian-accented English TTS
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="ishita"
            ),
        )

    async def on_enter(self):
        """Called when user joins - agent starts the conversation"""
        logger.info(f"Agent {self.__class__.__name__} entered the room")
        await self.session.generate_reply()


async def process_video_track(track):
    """Process a single video track (like a screen share)"""
    try:
        logger.info(f"Starting video stream for track: {track.sid}")
        video_stream = VideoStream(track)
        async for _frame in video_stream:
            # Proof of concept: we are receiving frames.
            # In a full vision implementation, you'd sample frames here.
            await asyncio.sleep(2.0)
    except Exception as e:
        logger.error(f"Error processing video track: {e}")


async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    logger.info(f"--- New Job: {ctx.job.id} ---")

    try:
        # CRITICAL: Connect first to initialize the room properly
        await ctx.connect()
        logger.info(f"Connected to room: {ctx.room.name}")

        # Use the room's event emitter to detect screen shares
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if publication.source == TrackSource.SCREEN_SHARE:
                logger.info(f"✅ Screen share detected from {participant.identity}!")
                # Start processing the video track in the background
                asyncio.create_task(process_video_track(track))
            elif track.kind == TrackKind.KIND_VIDEO:
                logger.info(f"Video track subscribed: {publication.source}")

        # Create and start the agent session (Voice part)
        session = AgentSession()
        await session.start(
            agent=ScreenAwareVoiceAgent(),
            room=ctx.room
        )
        logger.info("Voice session initialized and started")

        # Keep the entrypoint alive as long as the room connection exists
        # Note: 'isconnected' (no underscore) is the correct attribute in this SDK version
        while ctx.room.isconnected:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
    finally:
        logger.info(f"Job finished: {ctx.job.id}")


if __name__ == "__main__":
    # Check for required API keys
    required_keys = ["OPENAI_API_KEY", "SARVAM_API_KEY", "GROQ_API_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        # Note: We continue even if missing to allow the agent to fail gracefully in the worker
        # but in many dev setups you might want to exit(1)

    # Run the agent using the standard LiveKit CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
