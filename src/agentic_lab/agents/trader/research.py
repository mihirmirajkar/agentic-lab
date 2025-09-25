"""Advanced ticker research agent producing an analyst-style report.

Capabilities:
1. Frequency-aware lookback (driven by trading profile config) that tunes search horizon & depth.
2. Internet search via ddgs for multi-faceted queries (news, fundamentals, guidance, sentiment, macro, risk).
3. Optional market data snapshot (price performance & volume stats) if ``yfinance`` is available.
4. LLM-powered multi-section report (executive summary, catalysts, fundamentals, sentiment, risks, outlook).
5. Fallback heuristic summarization when LLM unavailable.

Configuration Sources:
    - Inline config dataclass parameters.
    - External YAML profile file (``configs/trading_profiles.yaml``) mapping trading frequency -> lookback_days & max_results.

Install extras:
    pip install 'agentic-lab[research]'        # LLM + search only
    pip install 'agentic-lab[research,trading]' # + market data via yfinance

Environment:
    OPENAI_API_KEY (optional) enables richer LLM summarization.

Safety Note: Network calls (search & LLM) are optional; module degrades gracefully when unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from agentic_lab.core.base import Agent
from agentic_lab.core.exceptions import AgentError
from agentic_lab.utils.logging import setup_logger

# Modular components
from agentic_lab.agents.trader.config import TickerResearchConfig, ProfileManager
from agentic_lab.agents.trader.market_data import MarketDataProvider
from agentic_lab.agents.trader.search import SearchProvider
from agentic_lab.agents.trader.scoring import ScoreProvider
from agentic_lab.agents.trader.reporting import ReportProvider


class TickerResearchAgent(Agent):
    """Agent that performs internet search & summarizes ticker context."""

    def __init__(self, config: TickerResearchConfig) -> None:
        super().__init__(config)
        self._logger = setup_logger(f"TickerResearchAgent[{config.ticker.upper()}]")
        self._step_counter = 0
        
        # Initialize modular components
        self._profile_manager = ProfileManager(config, self._logger)
        self._market_data_provider = MarketDataProvider(config, self._logger) 
        self._search_provider = SearchProvider(config, self._logger)
        self._score_provider = ScoreProvider(config, self._logger)
        self._report_provider = ReportProvider(config, self._logger)
        
        # Data storage
        self._raw_results: List[Dict[str, Any]] = []
        self._market_data: Optional[Any] = None
        self._quantitative_scores: Dict[str, float] = {}
        self._trading_signal: Optional[Dict[str, Any]] = None

    # ---------------------- Logging Helpers ---------------------- #
    def _step(self, message: str) -> None:
        self._step_counter += 1
        self._logger.info(f"[Step {self._step_counter}] {message}")

    @property
    def cfg(self) -> TickerResearchConfig:
        return self.config  # type: ignore[return-value]

    def initialize(self) -> None:  # type: ignore[override]
        self._step("Initializing agent")
        # Apply trading profile configuration
        self._profile_manager.apply_profile_if_needed()
        self._initialized = True
        self._step("Initialization complete")

    # ---------------------- Core workflow methods ---------------------- #
    def execute(self, ticker: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[override]
        """Execute the complete research workflow."""
        if not self.is_initialized():
            raise AgentError("TickerResearchAgent not initialized")
        
        if ticker and ticker != self.cfg.ticker:
            self.cfg.ticker = ticker
            # Update all components with new ticker
            self._profile_manager = ProfileManager(self.cfg, self._logger)
            self._market_data_provider = MarketDataProvider(self.cfg, self._logger)
            self._search_provider = SearchProvider(self.cfg, self._logger)
            self._score_provider = ScoreProvider(self.cfg, self._logger)
            self._report_provider = ReportProvider(self.cfg, self._logger)
        
        # Step 1: Apply profile configuration if needed
        self._profile_manager.apply_profile_if_needed()
        
        # Step 2: Fetch market data
        self._step("Fetching market data")
        self._market_data = self._market_data_provider.fetch_market_data()
        
        # Step 3: Build and execute search queries
        self._step("Building search queries")
        queries = self._search_provider.build_queries()
        
        self._step(f"Executing search across {len(queries)} query categories")
        self._raw_results = self._search_provider.search(queries)
        
        # Step 4: Compute quantitative scores
        self._step("Computing quantitative analysis")
        market_df = self._market_data.get("dataframe") if isinstance(self._market_data, dict) else self._market_data
        self._quantitative_scores = self._score_provider.calculate_all_scores(
            market_df, self._raw_results
        )
        
        # Step 5: Generate trading signal if enabled
        if self.cfg.enable_trading_signals:
            self._step("Generating trading recommendation")
            signal, confidence, reasoning = self._score_provider.generate_trading_signal(
                self._quantitative_scores["composite"],
                self._quantitative_scores["technical"],
                current_price=market_df.iloc[-1].get("Close") if market_df is not None and not market_df.empty else None
            )
            
            self._trading_signal = {
                "recommendation": signal,
                "confidence": round(confidence * 100, 1),
                "reasoning": reasoning,
            }
        
        # Step 6: Generate comprehensive report
        self._step("Generating comprehensive report")
        report_text = self._report_provider.generate_comprehensive_report(
            market_df,
            self._raw_results, 
            self._quantitative_scores,
            self._trading_signal["recommendation"] if self._trading_signal else "HOLD",
            self._trading_signal["confidence"] / 100 if self._trading_signal else 0.5,
            self._trading_signal["reasoning"] if self._trading_signal else "No trading signal generated"
        )
        
        self._step("Research workflow complete")
        
        # Return comprehensive results
        return {
            "ticker": self.cfg.ticker.upper(),
            "profile": self.cfg.trading_frequency,
            "lookback_days": self.cfg.lookback_days,
            "results_count": len(self._raw_results),
            "quantitative_scores": self._quantitative_scores,
            "trading_signal": self._trading_signal,
            "market_data_available": market_df is not None and not market_df.empty,
            "sources": self._raw_results,
            "full_report": report_text,
        }

    # ---------------------- Convenience methods ---------------------- #
    def get_last_summary(self) -> Optional[str]:
        """Get the executive summary from last execution."""
        return self._quantitative_scores.get("composite")

    def get_sources(self) -> List[Dict[str, Any]]:
        """Get the raw search results from last execution."""
        return list(self._raw_results)

    def get_trading_recommendation(self) -> Optional[Dict[str, Any]]:
        """Get the trading signal from last execution."""
        return self._trading_signal


__all__ = [
    "TickerResearchAgent",
    "TickerResearchConfig",
]


def research_ticker(ticker: str) -> None:
    # Demo run showcasing trading analysis (requires ddgs and optionally yfinance & OpenAI key)
    cfg = TickerResearchConfig(
        name="trading-analyst", 
        ticker=ticker, 
        trading_profile="quarterly"
    )
    agent = TickerResearchAgent(cfg)
    agent._logger.setLevel(logging.INFO)  # Show step-by-step progress
    
    try:
        agent.initialize()
        report = agent.execute()
        
        print(f"\n=== TRADING ANALYSIS for {report['ticker']} ===")
        print(f"Profile: {report.get('profile')} | Lookback: {report.get('lookback_days')} days")
        
        # Show key trading metrics
        if report.get('trading_signal'):
            signal = report['trading_signal']
            print(f"\n🎯 RECOMMENDATION: {signal['recommendation']} (Confidence: {signal['confidence']}%)")
            print(f"📊 Reasoning: {signal['reasoning']}")
        
        # Show quantitative scores
        if report.get('quantitative_scores'):
            scores = report['quantitative_scores']
            print(f"\n📈 QUANTITATIVE SCORES:")
            for name, score in scores.items():
                print(f"  {name.title()}: {score:.2f}")
        
        print(f"\n📰 Sources analyzed: {report.get('results_count', 0)}")
        print("\n" + "="*60)
        print(report.get('full_report', 'No report generated'))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    research_ticker("ARKK")

#TODO: the risk affects the final score and recommendation, its a feedback loop where changing risk from conservative to moderate gives the same result of holding. This needs to be fixed.