#!/usr/bin/env python
"""
LiveKit Context-Optimized Agent Launcher

This script sets up the Python path and launches the optimized agent.
Run from the project root: python run_livekit_agent.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from src.livekit_context_optimized import cli
    from livekit.agents import WorkerOptions
    from src.livekit_context_optimized import entrypoint

    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger("livekit-agent-launcher")
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
