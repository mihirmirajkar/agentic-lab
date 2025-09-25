"""Report generation and formatting for ticker research."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    import openai
except Exception:  # pragma: no cover - optional
    openai = None  # type: ignore

from agentic_lab.core.exceptions import AgentError
from .config import TickerResearchConfig


class ReportProvider:
    """Handles comprehensive report generation for ticker research."""
    
    def __init__(self, config: TickerResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._llm_client = openai.OpenAI(api_key=config.openai_api_key) if openai and config.openai_api_key else None
    
    def format_data_summary(
        self,
        df: Optional[Any],
        search_results: List[Dict[str, Any]],
        scores: Dict[str, float],
    ) -> str:
        """Format data summary section."""
        summary = [f"# {self.config.ticker.upper()} Research Summary"]
        
        # Market data summary
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            price = latest.get("Close")
            volume = latest.get("Volume")
            
            summary.append("\n## Market Data")
            if price:
                summary.append(f"- Current Price: ${price:.2f}")
            if volume:
                summary.append(f"- Volume: {volume:,.0f}")
            
            # Technical indicators
            rsi = latest.get("RSI")
            sma_20 = latest.get("SMA_20")
            macd = latest.get("MACD")
            
            if any(x is not None for x in [rsi, sma_20, macd]):
                summary.append("\n## Technical Indicators")
                if rsi is not None:
                    summary.append(f"- RSI: {rsi:.2f}")
                if sma_20 is not None:
                    summary.append(f"- SMA(20): ${sma_20:.2f}")
                if macd is not None:
                    summary.append(f"- MACD: {macd:.4f}")
        
        # Search results summary
        if search_results:
            summary.append(f"\n## Information Sources")
            summary.append(f"- Total sources analyzed: {len(search_results)}")
            
            categories = {}
            for result in search_results:
                cat = result.get("category", "general")
                categories[cat] = categories.get(cat, 0) + 1
            
            for cat, count in categories.items():
                summary.append(f"- {cat.title()} sources: {count}")
        
        # Quantitative scores
        summary.append(f"\n## Quantitative Analysis")
        for score_name, value in scores.items():
            summary.append(f"- {score_name.title()} Score: {value:.2f} ({self._score_interpretation(value)})")
        
        return "\n".join(summary)
    
    def _score_interpretation(self, score: float) -> str:
        """Convert numeric score to interpretation."""
        if score >= 0.7:
            return "Strong"
        elif score >= 0.5:
            return "Moderate"
        else:
            return "Weak"
    
    def format_trading_recommendation(
        self,
        signal: str,
        confidence: float,
        reasoning: str,
        current_price: Optional[float] = None,
    ) -> str:
        """Format trading recommendation section."""
        rec = ["\n## Trading Recommendation"]
        
        # Signal with confidence
        rec.append(f"**Signal: {signal}** (Confidence: {confidence:.1%})")
        rec.append(f"\n**Reasoning:** {reasoning}")
        
        # Risk assessment
        risk_level = self._assess_risk_level(confidence, signal)
        rec.append(f"\n**Risk Assessment:** {risk_level}")
        
        # Position sizing guidance
        position_guidance = self._position_sizing_guidance(signal, confidence)
        rec.append(f"\n**Position Sizing:** {position_guidance}")
        
        if current_price:
            rec.append(f"\n**Reference Price:** ${current_price:.2f}")
        
        return "\n".join(rec)
    
    def _assess_risk_level(self, confidence: float, signal: str) -> str:
        """Assess risk level based on confidence and signal."""
        if confidence >= 0.8:
            return f"Low risk - High confidence {signal} signal"
        elif confidence >= 0.6:
            return f"Medium risk - Moderate confidence {signal} signal"
        else:
            return f"High risk - Low confidence {signal} signal"
    
    def _position_sizing_guidance(self, signal: str, confidence: float) -> str:
        """Provide position sizing guidance."""
        risk_tolerance = self.config.risk_tolerance.lower()
        
        if signal == "BUY":
            if confidence >= 0.8:
                sizes = {"conservative": "Small position", "moderate": "Standard position", "aggressive": "Large position"}
            elif confidence >= 0.6:
                sizes = {"conservative": "Very small position", "moderate": "Small position", "aggressive": "Standard position"}
            else:
                sizes = {"conservative": "Avoid", "moderate": "Very small position", "aggressive": "Small position"}
        elif signal == "SELL":
            return "Consider reducing/closing positions"
        else:  # HOLD
            return "Maintain current position size"
        
        return sizes.get(risk_tolerance, "Standard position")
    
    def build_llm_prompt(
        self,
        search_results: List[Dict[str, Any]],
        scores: Dict[str, float],
        df: Optional[Any] = None,
    ) -> str:
        """Build comprehensive LLM prompt for analysis."""
        prompt_parts = [
            f"You are a professional financial analyst. Analyze {self.config.ticker.upper()} stock.",
            f"Trading Profile: {self.config.trading_frequency} ({self.config.risk_tolerance} risk tolerance)",
            f"Analysis Period: {self.config.lookback_days} days\n"
        ]
        
        # Market context
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            price = latest.get("Close")
            rsi = latest.get("RSI")
            
            prompt_parts.append("MARKET DATA:")
            if price:
                prompt_parts.append(f"Current Price: ${price:.2f}")
            if rsi is not None:
                prompt_parts.append(f"RSI: {rsi:.2f}")
        
        # Quantitative analysis
        prompt_parts.append(f"\nQUANTITATIVE SCORES:")
        for name, score in scores.items():
            prompt_parts.append(f"{name.title()}: {score:.2f}/1.0")
        
        # Information sources
        if search_results:
            prompt_parts.append(f"\nINFORMATION SOURCES ({len(search_results)} items):")
            for i, result in enumerate(search_results[:8], 1):  # Limit for prompt size
                title = result.get("title", "")[:100]
                snippet = result.get("snippet", "")[:200]
                category = result.get("category", "general")
                prompt_parts.append(f"{i}. [{category.upper()}] {title}")
                if snippet:
                    prompt_parts.append(f"   {snippet}...")
        
        prompt_parts.append(f"""
ANALYSIS REQUEST:
1. Synthesize all information into actionable insights
2. Assess key catalysts and risks for {self.config.trading_frequency} timeframe
3. Comment on sector/macro trends affecting the stock
4. Evaluate alignment between technical and fundamental factors
5. Provide specific trading considerations

Format your response with clear sections and bullet points.""")
        
        return "\n".join(prompt_parts)
    
    def generate_llm_analysis(self, prompt: str) -> Optional[str]:
        """Generate LLM-powered analysis."""
        if not self._llm_client:
            self.logger.warning("OpenAI client not configured - skipping LLM analysis")
            return None
        
        try:
            self.logger.info("Generating LLM analysis...")
            response = self._llm_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3,
            )
            
            analysis = response.choices[0].message.content
            self.logger.info("LLM analysis generated successfully")
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return None
    
    def generate_comprehensive_report(
        self,
        df: Optional[Any],
        search_results: List[Dict[str, Any]],
        scores: Dict[str, float],
        signal: str,
        confidence: float,
        reasoning: str,
    ) -> str:
        """Generate the complete research report."""
        self.logger.info("Assembling comprehensive report...")
        
        report_sections = []
        
        # Data summary
        report_sections.append(self.format_data_summary(df, search_results, scores))
        
        # Trading recommendation
        current_price = df.iloc[-1].get("Close") if df is not None and not df.empty else None
        report_sections.append(
            self.format_trading_recommendation(signal, confidence, reasoning, current_price)
        )
        
        # LLM analysis
        if self.config.enable_llm_analysis and self.config.openai_api_key:
            prompt = self.build_llm_prompt(search_results, scores, df)
            llm_analysis = self.generate_llm_analysis(prompt)
            
            if llm_analysis:
                report_sections.append("\n## Professional Analysis")
                report_sections.append(llm_analysis)
        
        # Source details
        if search_results and self.config.include_source_details:
            report_sections.append(self._format_source_details(search_results))
        
        # Configuration info
        report_sections.append(self._format_config_footer())
        
        self.logger.info("Report generation complete")
        return "\n\n".join(report_sections)
    
    def _format_source_details(self, search_results: List[Dict[str, Any]]) -> str:
        """Format detailed source information."""
        details = ["\n## Source Details"]
        
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "Untitled")[:80]
            url = result.get("url", "")
            category = result.get("category", "general")
            
            details.append(f"\n{i}. **[{category.upper()}]** {title}")
            if url:
                details.append(f"   Source: {url}")
        
        return "\n".join(details)
    
    def _format_config_footer(self) -> str:
        """Format configuration information footer."""
        return f"""
---
**Analysis Configuration:**
- Ticker: {self.config.ticker.upper()}
- Profile: {self.config.trading_frequency} ({self.config.risk_tolerance} risk)
- Lookback: {self.config.lookback_days} days
- Sources: {self.config.max_results} max
- Generated: {self.config.timestamp}"""


__all__ = ["ReportProvider"]