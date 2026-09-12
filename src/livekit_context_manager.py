"""
LiveKit Native Context Manager
Provides context compaction and summarization utilities for long-running voice agents.
"""

import logging
from typing import Optional

from livekit.agents import Agent, llm

logger = logging.getLogger("livekit-context-manager")
logger.setLevel(logging.INFO)


class ContextManager:
    """Helper for LiveKit-native context management"""

    @staticmethod
    async def get_compacted_context(
        agent: Agent,
        keep_last_n_turns: int = 10,
        exclude_system: bool = True
    ) -> llm.ChatContext:
        """
        Create a compacted version of agent's chat context.

        Filters out noise (system messages, function calls, config updates)
        and truncates to recent N turns if needed.

        Args:
            agent: The agent instance with chat_ctx attribute
            keep_last_n_turns: Number of recent turns to keep uncompressed
            exclude_system: Whether to exclude system/developer messages

        Returns:
            Filtered ChatContext ready for LLM calls

        Example:
            >>> compacted_ctx = await ContextManager.get_compacted_context(
            ...     agent=self,
            ...     keep_last_n_turns=10,
            ...     exclude_system=True
            ... )
            >>> await self.update_chat_ctx(compacted_ctx)
        """
        # Step 1: Copy with exclusions to remove noise
        compacted_ctx = agent.chat_ctx.copy(
            exclude_function_call=True,      # Remove tool calls/outputs
            exclude_instructions=True,       # Remove developer instructions
            exclude_config_update=True,      # Remove agent config updates
            exclude_handoff=True,            # Remove handoff messages
            exclude_empty_message=True,      # Remove empty messages
        )

        # Step 2: Truncate to recent N items if needed
        if len(compacted_ctx.items) > keep_last_n_turns:
            logger.info(
                f"Compacting context: {len(compacted_ctx.items)} items -> {keep_last_n_turns} items"
            )
            compacted_ctx.truncate(max_items=keep_last_n_turns)
            logger.info("✅ Context compaction complete")

        return compacted_ctx

    @staticmethod
    async def summarize_if_needed(
        agent: Agent,
        threshold_items: int = 20,
        llm_for_summary: Optional[llm.LLM] = None,
        keep_last_turns: int = 3
    ) -> llm.ChatContext:
        """
        Summarize context if approaching token limits.

        Uses LLM to compress old conversation history while keeping recent
        turns uncompressed for coherence.

        Args:
            agent: The agent instance
            threshold_items: Trigger summarization if item count exceeds this
            llm_for_summary: LLM instance for summarization (uses session.llm if None)
            keep_last_turns: Number of recent turns to keep uncompressed

        Returns:
            ChatContext (summarized if threshold exceeded, otherwise unchanged)

        Example:
            >>> await ContextManager.summarize_if_needed(
            ...     agent=self,
            ...     threshold_items=20,
            ...     keep_last_turns=3
            ... )
        """
        current_items = len(agent.chat_ctx.items)

        if current_items <= threshold_items:
            # No summarization needed
            return agent.chat_ctx

        # Need to summarize
        logger.info(f"Context has {current_items} items, summarizing...")

        # Use session's LLM for summarization if none provided
        if llm_for_summary is None:
            llm_for_summary = agent.session.llm

        # Create a copy with exclusions for summarization
        ctx_to_summarize = agent.chat_ctx.copy(
            exclude_function_call=True,
            exclude_instructions=True,
            exclude_config_update=True,
            exclude_empty_message=True,
        )

        try:
            # Use built-in _summarize() from ChatContext
            summary_ctx = await ctx_to_summarize._summarize(
                llm_v=llm_for_summary,
                keep_last_turns=keep_last_turns
            )

            # Replace agent's chat context with summarized version
            agent._chat_ctx = summary_ctx

            logger.info(
                f"✅ Summarization complete: {current_items} -> "
                f"{len(summary_ctx.items)} items"
            )
            return summary_ctx

        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            logger.info("Continuing with uncompacted context...")
            return agent.chat_ctx

    @staticmethod
    def get_context_stats(agent: Agent) -> dict:
        """
        Get statistics about current chat context.

        Useful for monitoring and debugging context growth.

        Args:
            agent: The agent instance

        Returns:
            Dictionary with context statistics

        Example:
            >>> stats = ContextManager.get_context_stats(self)
            >>> print(f"Items: {stats['total_items']}")
        """
        items = agent.chat_ctx.items

        # Count by type
        type_counts = {}
        for item in items:
            item_type = type(item).__name__
            type_counts[item_type] = type_counts.get(item_type, 0) + 1

        return {
            "total_items": len(items),
            "type_breakdown": type_counts,
            "has_system_messages": any(
                hasattr(item, 'type') and item.type == 'system' for item in items
            ),
            "has_function_calls": any(
                hasattr(item, 'type') and item.type == 'function-call' for item in items
            ),
        }
