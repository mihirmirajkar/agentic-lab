"""Base classes for agentic lab framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AgentConfig:
    """Base configuration for agents."""

    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = {}


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.config.name

    @property
    def description(self) -> Optional[str]:
        """Get the agent description."""
        return self.config.description

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent."""
        pass

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main functionality."""
        pass

    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized
