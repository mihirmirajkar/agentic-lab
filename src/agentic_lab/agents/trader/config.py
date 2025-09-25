"""Configuration classes and profile handling for ticker research."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pathlib

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from agentic_lab.core.base import AgentConfig
from agentic_lab.core.exceptions import AgentError


@dataclass
class TickerResearchConfig(AgentConfig):
    """Configuration for ticker research & report generation.

    Parameters
    ----------
    ticker: Stock symbol.
    trading_profile: Named profile referencing YAML (e.g. 'daily_1', 'weekly_1').
    profile_config_path: Path to YAML profiles file. Defaults to ``configs/trading_profiles.yaml``.
    lookback_days: Override for number of calendar days to analyze (if provided overrides profile value).
    max_results: Max aggregated search results (may be overridden by profile if not explicitly set).
    model: OpenAI model name.
    include_financial_terms: Whether to expand queries with fundamental terms.
    search_lang: Language code for search.
    extended_report: If True, produce multi-section analyst report.
    sections: Optional list controlling which sections to produce (default standard set).
    enable_trading_signals: If True, generate BUY/HOLD/SELL recommendations.
    include_technical_analysis: If True, compute and include technical indicators.
    enable_quantitative_scoring: If True, generate numerical scores for decision-making.
    """

    ticker: str = ""
    trading_profile: Optional[str] = None
    profile_config_path: Optional[str] = None
    lookback_days: Optional[int] = None
    max_results: int = 8
    model: str = "gpt-4o-mini"
    include_financial_terms: bool = True
    search_lang: str = "en"
    extended_report: bool = True
    sections: Optional[List[str]] = None
    enable_trading_signals: bool = True
    include_technical_analysis: bool = True
    enable_quantitative_scoring: bool = True
    enable_llm_analysis: bool = True
    include_source_details: bool = False

    # Derived / internal fields not set by user
    _profile_applied: bool = field(default=False, init=False, repr=False)
    
    @property
    def trading_frequency(self) -> Optional[str]:
        """Get the trading profile/frequency name."""
        return self.trading_profile
    
    @property
    def timestamp(self) -> str:
        """Get current timestamp for reporting."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @property 
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        import os
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def risk_tolerance(self) -> str:
        """Get risk tolerance from trading profile or default."""
        # This will be set by ProfileManager.apply_profile_if_needed()
        if hasattr(self, '_risk_tolerance'):
            return self._risk_tolerance
        return "moderate"  # Default fallback

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if not self.ticker:
            raise AgentError("TickerResearchConfig requires non-empty ticker")
        if self.max_results <= 0:
            raise AgentError("max_results must be > 0")
        if self.sections is None:
            self.sections = [
                "Executive Summary",
                "Company Overview",
                "Market & Price Action",
                "Technical Analysis",
                "Recent News & Catalysts",
                "Fundamentals Snapshot",
                "Sentiment & Analyst Tone",
                "Opportunities",
                "Risks",
                "Trading Recommendation",
                "Quantitative Scores",
                "Action Items",
            ]


class ProfileManager:
    """Handles trading profile loading and application."""
    
    def __init__(self, config: TickerResearchConfig, logger):
        self.config = config
        self.logger = logger
        self._profile_meta: Dict[str, Any] = {}
    
    def default_profile_path(self) -> pathlib.Path:
        """Resolve profile file path with robust fallbacks."""
        if self.config.profile_config_path:
            return pathlib.Path(self.config.profile_config_path)

        file_path = pathlib.Path(__file__).resolve()
        # Go up from trader/config.py to project root
        candidates = []
        try:
            project_root = file_path.parents[4]  # config.py -> trader -> agents -> agentic_lab -> src -> project_root
            candidates.append(project_root / "configs" / "trading_profiles.yaml")
        except IndexError:  # pragma: no cover
            pass
        
        # Additional fallbacks
        try:
            candidates.append(file_path.parents[3] / "configs" / "trading_profiles.yaml")
        except IndexError:  # pragma: no cover
            pass
        
        candidates.append(pathlib.Path.cwd() / "configs" / "trading_profiles.yaml")

        for c in candidates:
            if c.exists():
                return c
        return candidates[0]  # Return primary intended path for clear error
    
    def apply_profile_if_needed(self) -> None:
        """Apply trading profile settings to config if specified."""
        if self.config.trading_profile and not self.config._profile_applied:
            profiles_path = self.default_profile_path()
            if not profiles_path.exists():
                raise AgentError(f"Profile file not found: {profiles_path}")
            if yaml is None:
                raise AgentError("PyYAML not installed; required for profile loading")
            
            with profiles_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            profiles = (data.get("profiles") or {}) if isinstance(data, dict) else {}
            if self.config.trading_profile not in profiles:
                raise AgentError(f"Unknown trading_profile '{self.config.trading_profile}'")
            
            profile = profiles[self.config.trading_profile]
            # Only apply if user did not manually set
            if self.config.lookback_days is None:
                self.config.lookback_days = int(profile.get("lookback_days", 21))
            if not self.config.max_results or self.config.max_results == 8:
                self.config.max_results = int(profile.get("max_results", self.config.max_results))
            # Set risk tolerance from profile (always comes from profile)
            self.config._risk_tolerance = profile.get("risk_tolerance", "moderate")
            
            self._profile_meta = profile
            self.config._profile_applied = True
        
        # Provide defaults if still unset
        if self.config.lookback_days is None:
            self.config.lookback_days = 21
        # Set default risk tolerance if no profile was applied
        if not hasattr(self.config, '_risk_tolerance'):
            self.config._risk_tolerance = "moderate"
    
    @property
    def profile_meta(self) -> Dict[str, Any]:
        """Get loaded profile metadata."""
        return self._profile_meta


__all__ = ["TickerResearchConfig", "ProfileManager"]