"""
Specialized Domain State Management
Stores persistent domain-specific state outside of chat context to reduce token usage.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SpecializedDomainState:
    """
    Persistent state for specialized task domain.

    Stores user preferences, task history, and domain facts outside
    the chat context to avoid consuming tokens.

    Example:
        >>> state = SpecializedDomainState()
        >>> state.set_user_preference("language", "en-IN")
        >>> state.add_task_result("data_analysis", {"rows": 100})
    """

    # User profile information (persists across sessions)
    user_profile: dict = field(default_factory=dict)

    # History of completed tasks (for reference)
    task_history: list[dict] = field(default_factory=list)

    # Domain-specific facts learned during conversation
    domain_facts: dict = field(default_factory=dict)

    # User preferences (language, voice, etc.)
    preferences: dict = field(default_factory=dict)

    # Session metadata
    session_start_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)

    def add_task_result(self, task_name: str, result: dict) -> None:
        """
        Store task completion result without adding to chat context.

        Args:
            task_name: Name of the completed task
            result: Dictionary containing task result data
        """
        self.task_history.append({
            "task": task_name,
            "timestamp": time.time(),
            "result": result
        })
        self.last_activity_time = time.time()

    def get_task_history(self, task_name: Optional[str] = None) -> list[dict]:
        """
        Retrieve task history.

        Args:
            task_name: Optional filter for specific task name

        Returns:
            List of task results (all tasks or filtered by name)
        """
        if task_name:
            return [t for t in self.task_history if t["task"] == task_name]
        return self.task_history

    def get_user_preference(self, key: str) -> Optional[str]:
        """
        Retrieve user preference from external state.

        Args:
            key: Preference key to retrieve

        Returns:
            Preference value or None if not set
        """
        return self.preferences.get(key)

    def set_user_preference(self, key: str, value: str) -> None:
        """
        Store user preference in external state.

        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.last_activity_time = time.time()

    def set_domain_fact(self, fact_key: str, fact_value: Any) -> None:
        """
        Store a domain-specific fact learned during conversation.

        Args:
            fact_key: Identifier for the fact
            fact_value: Fact data (can be any type)
        """
        self.domain_facts[fact_key] = fact_value
        self.last_activity_time = time.time()

    def get_domain_fact(self, fact_key: str) -> Optional[Any]:
        """
        Retrieve a domain-specific fact.

        Args:
            fact_key: Identifier for the fact

        Returns:
            Fact value or None if not found
        """
        return self.domain_facts.get(fact_key)

    def update_user_profile(self, profile_data: dict) -> None:
        """
        Update user profile with new information.

        Args:
            profile_data: Dictionary of profile attributes to update
        """
        self.user_profile.update(profile_data)
        self.last_activity_time = time.time()

    def get_session_duration(self) -> float:
        """
        Get session duration in seconds.

        Returns:
            Seconds since session start
        """
        return time.time() - self.session_start_time

    def get_idle_time(self) -> float:
        """
        Get idle time since last activity.

        Returns:
            Seconds since last activity
        """
        return time.time() - self.last_activity_time

    def to_summary(self) -> str:
        """
        Generate a text summary of the domain state.

        Useful for including in summarization prompts.

        Returns:
            String summary of key state information
        """
        parts = []

        if self.preferences:
            prefs_str = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            parts.append(f"User Preferences: {prefs_str}")

        if self.user_profile:
            profile_str = ", ".join(f"{k}={v}" for k, v in self.user_profile.items())
            parts.append(f"User Profile: {profile_str}")

        if self.domain_facts:
            facts_count = len(self.domain_facts)
            parts.append(f"Domain Facts Learned: {facts_count}")

        if self.task_history:
            tasks_count = len(self.task_history)
            parts.append(f"Tasks Completed: {tasks_count}")

        return "; ".join(parts) if parts else "No domain state accumulated yet."
