"""Test configuration and fixtures."""

import pytest

from agentic_lab.agents.trading import TradingConfig
from agentic_lab.core.base import AgentConfig


@pytest.fixture
def basic_agent_config():
    """Basic agent configuration for testing."""
    return AgentConfig(name="test-agent", description="A test agent")


@pytest.fixture
def trading_config():
    """Trading agent configuration for testing."""
    return TradingConfig(
        name="test-trading-agent",
        description="A test trading agent",
        symbols=["AAPL", "MSFT"],
        strategy="test_strategy",
        risk_tolerance=0.1,
    )
