"""Tests for TickerResearchAgent (offline / minimal paths).

These tests only validate local logic & error handling. They do NOT perform
real web calls; if duckduckgo-search is not installed, we expect an error on execute.
"""
import os
import pytest

from agentic_lab.agents.trader import (
    TickerResearchAgent,
    TickerResearchConfig,
)
from agentic_lab.core.exceptions import AgentError


def test_research_config_validation():
    with pytest.raises(AgentError):
        TickerResearchConfig(name="researcher", ticker="")

    cfg = TickerResearchConfig(name="researcher", ticker="AAPL")
    assert cfg.ticker == "AAPL"
    assert cfg.max_results == 8


def test_research_agent_init_and_initialize():
    cfg = TickerResearchConfig(name="researcher", ticker="MSFT", max_results=3)
    agent = TickerResearchAgent(cfg)
    assert not agent.is_initialized()
    agent.initialize()
    assert agent.is_initialized()


def test_profile_application(monkeypatch, tmp_path):
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("PyYAML not installed; skipping profile test")
    # Create a temporary profiles YAML
    content = """
profiles:
  test_profile:
    lookback_days: 10
    max_results: 5
default_profile: test_profile
"""
    p = tmp_path / "profiles.yaml"
    p.write_text(content, encoding="utf-8")
    cfg = TickerResearchConfig(
        name="researcher", ticker="AAPL", trading_profile="test_profile", profile_config_path=str(p)
    )
    agent = TickerResearchAgent(cfg)
    agent.initialize()
    assert cfg.lookback_days == 10
    assert cfg.max_results == 5


def test_extended_report_heuristic(monkeypatch):
    cfg = TickerResearchConfig(name="researcher", ticker="NFLX", max_results=2, extended_report=True)
    agent = TickerResearchAgent(cfg)
    agent.initialize()
    # Force no search backend & expect error
    agent._search_backend = None  # type: ignore
    with pytest.raises(AgentError):
        agent.execute()


@pytest.mark.parametrize("installed", [False])
def test_execute_without_search_backend(monkeypatch, installed):
    # Force DDGS absence
    monkeypatch.setenv("OPENAI_API_KEY", "")
    cfg = TickerResearchConfig(name="researcher", ticker="GOOGL", max_results=2)
    agent = TickerResearchAgent(cfg)
    agent.initialize()

    # Monkeypatch _search_backend to None to simulate missing dependency
    agent._search_backend = None  # type: ignore
    with pytest.raises(AgentError, match="ddgs not installed"):
        agent.execute()
