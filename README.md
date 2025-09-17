# agentic-lab

A playground for building, experimenting, and scaling agentic systems. Starting with a trading agent, expanding to multi-domain autonomous workflows.

## Project Structure

This project follows a clean and modern Python package structure:

```
agentic-lab/
├── src/agentic_lab/          # Main package source code
│   ├── __init__.py           # Package initialization
│   ├── core/                 # Core framework components
│   │   ├── __init__.py
│   │   ├── base.py           # Base classes for agents
│   │   └── exceptions.py     # Custom exceptions
│   ├── agents/               # Specialized agent implementations
│   │   ├── __init__.py
│   │   └── trading.py        # Trading agent implementation
│   └── utils/                # Utility functions and helpers
│       ├── __init__.py
│       ├── logging.py        # Logging utilities
│       └── validators.py     # Validation functions
├── tests/                    # Test suite
│   ├── conftest.py          # Test configuration and fixtures
│   ├── test_package.py      # Package-level tests
│   ├── unit/                # Unit tests
│   │   ├── test_core_base.py
│   │   ├── test_core_exceptions.py
│   │   ├── test_agents_trading.py
│   │   └── test_utils.py
│   └── integration/         # Integration tests
│       └── test_workflow.py
├── pyproject.toml           # Modern Python packaging configuration
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Features

- **Modern Python Structure**: Uses `src/` layout with proper package organization
- **Extensible Agent Framework**: Base classes for building different types of agents
- **Trading Agent**: Initial implementation for trading and financial workflows
- **Comprehensive Testing**: Unit and integration tests with pytest
- **Type Hints**: Full type annotation support
- **Development Tools**: Configured for black, isort, flake8, and mypy
- **Modern Packaging**: Uses pyproject.toml for configuration

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/mihirmirajkar/agentic-lab.git
cd agentic-lab

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Production Installation

```bash
pip install agentic-lab
```

## Quick Start

```python
from agentic_lab.agents.trading import TradingAgent, TradingConfig

# Create a trading agent configuration
config = TradingConfig(
    name="my-trading-agent",
    description="A sample trading agent",
    symbols=["AAPL", "MSFT", "GOOGL"],
    strategy="buy_and_hold",
    risk_tolerance=0.05
)

# Create and initialize the agent
agent = TradingAgent(config)
agent.initialize()

# Execute trading actions
result = agent.execute("analyze", "AAPL")
print(result)

# Get portfolio summary
summary = agent.get_portfolio_summary()
print(f"Managing {summary['symbols_count']} symbols")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/agentic_lab

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture

The framework is built around a few core concepts:

### Agent Base Classes

- **`Agent`**: Abstract base class for all agents
- **`AgentConfig`**: Configuration class for agent setup

### Specialized Agents

- **`TradingAgent`**: Implements trading-specific functionality
- **`TradingConfig`**: Configuration for trading agents

### Utilities

- **Logging**: Consistent logging setup across the framework
- **Validation**: Common validation functions
- **Exceptions**: Custom exception hierarchy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
