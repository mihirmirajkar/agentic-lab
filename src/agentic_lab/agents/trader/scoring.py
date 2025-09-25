"""Scoring and trading signal generation for ticker research."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import logging

from .config import TickerResearchConfig


class ScoreProvider:
    """Handles quantitative scoring and trading signal generation."""
    
    def __init__(self, config: TickerResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def technical_score(self, df: Optional[Any]) -> float:
        """Compute weighted technical analysis score (0-1)."""
        if df is None or df.empty:
            return 0.5
        
        row = df.iloc[-1]
        score_components = []
        
        # RSI scoring (0-100 -> normalized)
        rsi = row.get("RSI")
        if rsi is not None:
            rsi_score = self._rsi_score_normalized(rsi)
            score_components.append(("RSI", rsi_score))
        
        # Price vs SMA scoring
        price = row.get("Close")
        sma_20 = row.get("SMA_20")
        if price is not None and sma_20 is not None:
            price_sma_score = min(max(price / sma_20, 0.7), 1.3) - 0.7
            price_sma_score = price_sma_score / 0.6  # normalize to 0-1
            score_components.append(("Price_vs_SMA", price_sma_score))
        
        # MACD scoring
        macd = row.get("MACD")
        macd_signal = row.get("MACD_Signal")
        if macd is not None and macd_signal is not None:
            macd_score = 1.0 if macd > macd_signal else 0.3
            score_components.append(("MACD", macd_score))
        
        if not score_components:
            return 0.5
        
        # Weighted average
        weights = {"RSI": 0.4, "Price_vs_SMA": 0.4, "MACD": 0.2}
        total_weight = sum(weights.get(name, 1.0) for name, _ in score_components)
        weighted_sum = sum(weights.get(name, 1.0) * score for name, score in score_components)
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        return max(0.0, min(1.0, final_score))
    
    def _rsi_score_normalized(self, rsi: float) -> float:
        """Convert RSI to normalized score (0-1) - more sensitive to extremes."""
        if rsi <= 25:
            return 0.9  # Very oversold = strong buy signal
        elif rsi <= 30:
            return 0.75  # Oversold = buy opportunity
        elif rsi >= 75:
            return 0.15  # Very overbought = strong sell signal
        elif rsi >= 70:
            return 0.25  # Overbought = sell signal
        else:
            # More aggressive scaling in the middle range
            return 0.75 - (rsi - 30) * 0.5 / 40
    
    def momentum_score(self, df: Optional[Any]) -> float:
        """Compute momentum score based on price movement (0-1)."""
        if df is None or len(df) < 2:
            return 0.5
        
        recent_price = df.iloc[-1].get("Close")
        older_price = df.iloc[-min(5, len(df))].get("Close")
        
        if recent_price is None or older_price is None:
            return 0.5
        
        pct_change = (recent_price - older_price) / older_price
        # More aggressive scaling: -15% = 0, +15% = 1, 0% = 0.5
        momentum = 0.5 + (pct_change * 3.33)
        return max(0.0, min(1.0, momentum))
    
    def sentiment_score(self, search_results: List[Dict[str, Any]]) -> float:
        """Compute sentiment score using GPT analysis of search results (0-1)."""
        if not search_results:
            return 0.5
        
        # Try GPT-based sentiment analysis first
        try:
            return self._gpt_sentiment_analysis(search_results)
        except Exception as e:
            self.logger.warning(f"GPT sentiment analysis failed: {e}. Falling back to neutral.")
            # Return neutral sentiment (0.5) when GPT is unavailable
            return 0.5
    
    def _gpt_sentiment_analysis(self, search_results: List[Dict[str, Any]]) -> float:
        """Use GPT to analyze sentiment from search results."""
        import os
        
        # Check if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OpenAI API key not available")
        
        try:
            from openai import OpenAI
        except ImportError:
            raise Exception("OpenAI package not installed")
        
        client = OpenAI(api_key=api_key)
        
        # Prepare text for analysis (limit to top 10 results to avoid token limits)
        texts = []
        for result in search_results[:10]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            if title or snippet:
                texts.append(f"Title: {title}\nContent: {snippet}")
        
        if not texts:
            return 0.5
        
        combined_text = "\n\n---\n\n".join(texts)
        
        prompt = f"""Analyze the sentiment of the following news articles and content about stock ticker {self.config.ticker.upper()}.

Rate the overall sentiment on a scale from 0 to 100 where:
- 0-20: Very Bearish (strong sell signals, major negative news)
- 21-40: Bearish (negative outlook, concerning developments)  
- 41-60: Neutral (mixed signals, no clear direction)
- 61-80: Bullish (positive outlook, good developments)
- 81-100: Very Bullish (strong buy signals, major positive news)

Focus on:
- Business fundamentals and outlook
- Financial performance and guidance
- Market position and competitive threats
- Management commentary and analyst opinions
- Recent developments and catalysts

Content to analyze:
{combined_text}

Respond with ONLY the numerical score (0-100), no explanation."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1  # Low temperature for consistent scoring
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Normalize from 0-100 to 0-1
            normalized_score = max(0.0, min(1.0, score / 100.0))
            
            self.logger.info(f"GPT sentiment analysis: {score}/100 -> {normalized_score:.2f}")
            return normalized_score
            
        except Exception as e:
            raise Exception(f"GPT API call failed: {e}")
    
    def composite_score(
        self,
        technical: float,
        momentum: float,
        sentiment: float,
        risk_adjustment: float = 1.0,
    ) -> float:
        """Compute weighted composite score with risk adjustment (0-1)."""
        
        # Check if sentiment is neutral (0.5) indicating GPT was unavailable
        use_sentiment = abs(sentiment - 0.5) > 0.01  # Allow small tolerance
        
        if use_sentiment:
            # Base weights when sentiment is available
            weights = {
                "technical": 0.4,
                "momentum": 0.3,
                "sentiment": 0.3,
            }
            
            # Dynamic weight adjustment: when both technical and momentum are weak,
            # reduce sentiment influence (prevents false positives from news sentiment)
            if technical < 0.5 and momentum < 0.4:
                weights["technical"] = 0.45
                weights["momentum"] = 0.35  
                weights["sentiment"] = 0.20  # Reduce sentiment weight when technicals are bearish
            
            composite = (
                weights["technical"] * technical +
                weights["momentum"] * momentum +
                weights["sentiment"] * sentiment
            )
        else:
            # When sentiment is not available, use only technical and momentum
            weights = {
                "technical": 0.6,  # Increased weight
                "momentum": 0.4,   # Increased weight
            }
            
            composite = (
                weights["technical"] * technical +
                weights["momentum"] * momentum
            )
            
            self.logger.info(f"Sentiment unavailable, using technical/momentum only")
        
        # Apply risk tolerance adjustment
        adjusted = composite * risk_adjustment
        return max(0.0, min(1.0, adjusted))
    
    def generate_trading_signal(
        self,
        composite_score: float,
        technical_score: float,
        current_price: Optional[float] = None,
    ) -> Tuple[str, float, str]:
        """
        Generate trading signal with confidence and reasoning.
        
        Returns:
            Tuple of (signal, confidence, reasoning)
        """
        risk_tolerance = self.config.risk_tolerance.lower()
        
        # Determine thresholds based on risk tolerance (made more responsive)
        if risk_tolerance == "conservative":
            buy_threshold = 0.60  # was 0.75
            sell_threshold = 0.40  # was 0.25
        elif risk_tolerance == "aggressive":
            buy_threshold = 0.50  # was 0.55  
            sell_threshold = 0.40  # was 0.35 (fixed: was incorrectly 0.45)
        else:  # moderate
            buy_threshold = 0.55  # was 0.65
            sell_threshold = 0.45  # was 0.30
        
        confidence = abs(composite_score - 0.5) * 2  # 0.0 to 1.0
        
        if composite_score >= buy_threshold:
            signal = "BUY"
            reasoning = f"Strong positive signals (score={composite_score:.2f}) exceed {risk_tolerance} buy threshold"
        elif composite_score <= sell_threshold:
            signal = "SELL"
            reasoning = f"Weak signals (score={composite_score:.2f}) below {risk_tolerance} sell threshold"
        else:
            signal = "HOLD"
            reasoning = f"Mixed signals (score={composite_score:.2f}) suggest maintaining position"
        
        # Adjust confidence based on technical score alignment
        if technical_score is not None:
            if (signal == "BUY" and technical_score > 0.6) or \
               (signal == "SELL" and technical_score < 0.4) or \
               (signal == "HOLD"):
                confidence = min(confidence * 1.2, 1.0)
            else:
                confidence *= 0.8
        
        return signal, confidence, reasoning
    
    def calculate_all_scores(
        self,
        df: Optional[Any],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate all scoring components."""
        technical = self.technical_score(df)
        momentum = self.momentum_score(df)
        sentiment = self.sentiment_score(search_results)
        
        # Risk adjustment based on profile
        risk_multipliers = {
            "conservative": 0.85,
            "moderate": 1.0,
            "aggressive": 1.15,
        }
        risk_adjustment = risk_multipliers.get(self.config.risk_tolerance.lower(), 1.0)
        
        composite = self.composite_score(technical, momentum, sentiment, risk_adjustment)
        
        return {
            "technical": technical,
            "momentum": momentum,
            "sentiment": sentiment,
            "composite": composite,
        }


__all__ = ["ScoreProvider"]