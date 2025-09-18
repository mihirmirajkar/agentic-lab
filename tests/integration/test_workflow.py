"""Integration tests for the agentic lab framework."""

import pytest

from agentic_lab.agents.trading import TradingAgent, TradingConfig
from agentic_lab.utils.logging import setup_logger


def test_end_to_end_trading_workflow():
    """Test complete trading agent workflow."""
    # Setup logging
    logger = setup_logger("integration_test")
    logger.info("Starting end-to-end trading workflow test")

    # Create trading configuration
    config = TradingConfig(
        name="integration-test-trader",
        description="Trading agent for integration testing",
        symbols=["AAPL", "MSFT", "GOOGL"],
        strategy="test_integration",
        risk_tolerance=0.05,
    )

    # Create and initialize agent
    agent = TradingAgent(config)
    agent.initialize()

    # Verify initialization
    assert agent.is_initialized()
    assert len(agent.positions) == 3

    # Execute some trading actions
    buy_result = agent.execute("buy", "AAPL", 10)
    sell_result = agent.execute("sell", "MSFT", 5)
    analyze_result = agent.execute("analyze", "GOOGL")

    # Verify results
    assert buy_result["action"] == "buy"
    assert buy_result["symbol"] == "AAPL"
    assert sell_result["action"] == "sell"
    assert analyze_result["action"] == "analyze"

    # Get portfolio summary
    summary = agent.get_portfolio_summary()
    assert summary["symbols_count"] == 3
    assert "positions" in summary

    logger.info("End-to-end trading workflow test completed successfully")


def test_error_handling_integration():
    """Test error handling across modules."""
    from agentic_lab.core.exceptions import AgentError

    # Test uninitialized agent
    config = TradingConfig(name="error-test", symbols=["AAPL"])
    agent = TradingAgent(config)

    with pytest.raises(AgentError):
        agent.execute("buy", "AAPL", 10)

    # Test with empty symbols
    empty_config = TradingConfig(name="empty-test", symbols=[])
    empty_agent = TradingAgent(empty_config)

    with pytest.raises(AgentError):
        empty_agent.initialize()
