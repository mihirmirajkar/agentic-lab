"""Core exceptions for agentic lab."""


class AgenticLabError(Exception):
    """Base exception for all agentic lab errors."""

    def __init__(self, message: str, code: str = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class AgentError(AgenticLabError):
    """Exception raised when agent operations fail."""

    pass


class ConfigurationError(AgenticLabError):
    """Exception raised when there are configuration issues."""

    pass


class ValidationError(AgenticLabError):
    """Exception raised when validation fails."""

    pass
