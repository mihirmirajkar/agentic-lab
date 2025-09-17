"""Tests for core exceptions."""

from agentic_lab.core.exceptions import (
    AgentError,
    AgenticLabError,
    ConfigurationError,
    ValidationError,
)


def test_agentic_lab_error():
    """Test base AgenticLabError exception."""
    error = AgenticLabError("Test message")
    assert str(error) == "Test message"
    assert error.message == "Test message"
    assert error.code is None


def test_agentic_lab_error_with_code():
    """Test AgenticLabError with error code."""
    error = AgenticLabError("Test message", code="TEST001")
    assert error.code == "TEST001"


def test_agent_error():
    """Test AgentError exception."""
    error = AgentError("Agent failed")
    assert str(error) == "Agent failed"
    assert isinstance(error, AgenticLabError)


def test_configuration_error():
    """Test ConfigurationError exception."""
    error = ConfigurationError("Config invalid")
    assert str(error) == "Config invalid"
    assert isinstance(error, AgenticLabError)


def test_validation_error():
    """Test ValidationError exception."""
    error = ValidationError("Validation failed")
    assert str(error) == "Validation failed"
    assert isinstance(error, AgenticLabError)
