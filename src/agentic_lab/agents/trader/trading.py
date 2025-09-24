"""Trading agent implementation."""

from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, List, Optional

from agentic_lab.core.base import Agent, AgentConfig
from agentic_lab.core.exceptions import AgentError


@dataclass
class TradingConfig(AgentConfig):
    """Configuration for trading agent."""

    symbols: List[str] = None
    strategy: str = "buy_and_hold"
    risk_tolerance: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.symbols is None:
            self.symbols = []


class TradingAgent(Agent):
    """Agent for trading operations and strategies."""

    def __init__(self, config: TradingConfig) -> None:
        super().__init__(config)
        self.symbols = config.symbols
        self.strategy = config.strategy
        self.risk_tolerance = config.risk_tolerance
        self.positions: Dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize the trading agent."""
        if not self.symbols:
            raise AgentError("No symbols configured for trading agent")

        # Initialize positions tracking
        for symbol in self.symbols:
            self.positions[symbol] = {
                "quantity": 0,
                "avg_price": 0.0,
                "total_value": 0.0,
            }

        self._initialized = True

    def execute(
        self, action: str, symbol: str, quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute trading actions."""
        if not self.is_initialized():
            raise AgentError("Trading agent not initialized")

        if symbol not in self.symbols:
            raise AgentError(f"Symbol {symbol} not in configured symbols")

        if action not in ["buy", "sell", "analyze"]:
            raise AgentError(f"Unknown action: {action}")

        # This is a placeholder implementation
        # In a real implementation, this would connect to trading APIs
        result = {
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "status": "simulated",
            "message": f"Simulated {action} for {symbol}",
        }

        return result

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            "positions": self.positions.copy(),
            "total_value": sum(pos["total_value"] for pos in self.positions.values()),
            "symbols_count": len(self.symbols),
        }

def main() -> None:
    """Run a demo workflow with the simulated trading agent."""
    config = TradingConfig(
        name="demo-trader",
        description="Sample trading agent run",
        symbols=["AAPL", "MSFT", "GOOGL"],
        strategy="demo_strategy",
        risk_tolerance=0.05,
    )

    agent = TradingAgent(config)

    try:
        agent.initialize()
    except AgentError as exc:
        print(f"Initialization failed: {exc}")
        return

    for action, symbol, qty in [("buy", "AAPL", 10), ("sell", "MSFT", 5), ("analyze", "GOOGL", None)]:
        result = agent.execute(action, symbol, qty)
        print(f"{result['action'].title()} {result['symbol']} -> {result['status']}")

    summary = agent.get_portfolio_summary()
    print("Portfolio summary:")
    pprint(summary)
    


if __name__ == "__main__":
    main()