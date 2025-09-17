"""Tests for trading agent."""

import pytest

from agentic_lab.agents.trading import TradingAgent, TradingConfig
from agentic_lab.core.exceptions import AgentError


def test_trading_config_creation():
    """Test TradingConfig creation and defaults."""
    config = TradingConfig(name="test-trader")
    assert config.name == "test-trader"
    assert config.symbols == []
    assert config.strategy == "buy_and_hold"
    assert config.risk_tolerance == 0.05


def test_trading_config_with_symbols():
    """Test TradingConfig with symbols."""
    config = TradingConfig(
        name="test-trader",
        symbols=["AAPL", "MSFT"],
        strategy="momentum",
        risk_tolerance=0.1,
    )
    assert config.symbols == ["AAPL", "MSFT"]
    assert config.strategy == "momentum"
    assert config.risk_tolerance == 0.1


def test_trading_agent_creation(trading_config):
    """Test TradingAgent creation."""
    agent = TradingAgent(trading_config)
    assert agent.name == "test-trading-agent"
    assert agent.symbols == ["AAPL", "MSFT"]
    assert agent.strategy == "test_strategy"
    assert agent.risk_tolerance == 0.1
    assert not agent.is_initialized()


def test_trading_agent_initialization_success(trading_config):
    """Test successful trading agent initialization."""
    agent = TradingAgent(trading_config)
    agent.initialize()

    assert agent.is_initialized()
    assert "AAPL" in agent.positions
    assert "MSFT" in agent.positions
    assert agent.positions["AAPL"]["quantity"] == 0


def test_trading_agent_initialization_failure():
    """Test trading agent initialization failure with no symbols."""
    config = TradingConfig(name="test-trader", symbols=[])
    agent = TradingAgent(config)

    with pytest.raises(AgentError, match="No symbols configured"):
        agent.initialize()


def test_trading_agent_execute_buy(trading_config):
    """Test trading agent buy execution."""
    agent = TradingAgent(trading_config)
    agent.initialize()

    result = agent.execute("buy", "AAPL", 10)
    assert result["action"] == "buy"
    assert result["symbol"] == "AAPL"
    assert result["quantity"] == 10
    assert result["status"] == "simulated"


def test_trading_agent_execute_uninitialized(trading_config):
    """Test trading agent execution when not initialized."""
    agent = TradingAgent(trading_config)

    with pytest.raises(AgentError, match="Trading agent not initialized"):
        agent.execute("buy", "AAPL", 10)


def test_trading_agent_execute_invalid_symbol(trading_config):
    """Test trading agent execution with invalid symbol."""
    agent = TradingAgent(trading_config)
    agent.initialize()

    with pytest.raises(AgentError, match="Symbol GOOGL not in configured symbols"):
        agent.execute("buy", "GOOGL", 10)


def test_trading_agent_execute_invalid_action(trading_config):
    """Test trading agent execution with invalid action."""
    agent = TradingAgent(trading_config)
    agent.initialize()

    with pytest.raises(AgentError, match="Unknown action: invalid"):
        agent.execute("invalid", "AAPL", 10)


def test_trading_agent_portfolio_summary(trading_config):
    """Test trading agent portfolio summary."""
    agent = TradingAgent(trading_config)
    agent.initialize()

    summary = agent.get_portfolio_summary()
    assert "positions" in summary
    assert "total_value" in summary
    assert "symbols_count" in summary
    assert summary["symbols_count"] == 2
    assert summary["total_value"] == 0.0
