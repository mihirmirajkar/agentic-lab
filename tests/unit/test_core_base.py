"""Tests for core base classes."""

from agentic_lab.core.base import Agent, AgentConfig


class TestAgent(Agent):
    """Test implementation of Agent for testing."""

    def initialize(self):
        self._initialized = True

    def execute(self, *args, **kwargs):
        return {"result": "test executed", "args": args, "kwargs": kwargs}


def test_agent_config_creation(basic_agent_config):
    """Test AgentConfig creation and defaults."""
    assert basic_agent_config.name == "test-agent"
    assert basic_agent_config.description == "A test agent"
    assert basic_agent_config.config == {}


def test_agent_config_with_custom_config():
    """Test AgentConfig with custom config dictionary."""
    config = AgentConfig(
        name="test", description="test agent", config={"param1": "value1", "param2": 42}
    )
    assert config.config["param1"] == "value1"
    assert config.config["param2"] == 42


def test_agent_creation_and_properties(basic_agent_config):
    """Test Agent creation and property access."""
    agent = TestAgent(basic_agent_config)
    assert agent.name == "test-agent"
    assert agent.description == "A test agent"
    assert not agent.is_initialized()


def test_agent_initialization(basic_agent_config):
    """Test agent initialization."""
    agent = TestAgent(basic_agent_config)
    assert not agent.is_initialized()

    agent.initialize()
    assert agent.is_initialized()


def test_agent_execution(basic_agent_config):
    """Test agent execution."""
    agent = TestAgent(basic_agent_config)
    agent.initialize()

    result = agent.execute("arg1", "arg2", key1="value1")
    assert result["result"] == "test executed"
    assert result["args"] == ("arg1", "arg2")
    assert result["kwargs"] == {"key1": "value1"}
