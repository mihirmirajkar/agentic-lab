"""Market data fetching and technical analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

try:
    import yfinance as yf  # type: ignore
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore
    pd = None  # type: ignore

from agentic_lab.agents.trader.stock_analysis.analyzer import (
    calculate_moving_average,
    calculate_rsi,
    calculate_macd,
)
from .config import TickerResearchConfig


class MarketDataProvider:
    """Handles market data fetching and technical analysis computation."""
    
    def __init__(self, config: TickerResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.technical_indicators: Dict[str, Any] = {}
    
    def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data with extended history for technical analysis."""
        if yf is None:
            return {"available": False, "reason": "yfinance not installed"}
        
        try:
            self.logger.info("Fetching market data via yfinance")
            lookback = self.config.lookback_days or 21
            # Get more data for technical analysis if enabled
            fetch_days = max(lookback + 50, 90) if self.config.include_technical_analysis else max(lookback, 5)
            
            ticker_obj = yf.Ticker(self.config.ticker)
            hist = ticker_obj.history(period=f"{fetch_days}d")
            if hist.empty:
                self.logger.warning("Market data history is empty")
                return {"available": False, "reason": "empty history"}
            
            closes = hist["Close"].dropna()
            volume = hist["Volume"].dropna()
            highs = hist["High"].dropna()
            lows = hist["Low"].dropna()
            
            # Basic performance metrics
            pct_change_total = ((closes.iloc[-1] / closes.iloc[0]) - 1) * 100 if len(closes) > 1 else 0
            last_close = float(closes.iloc[-1])
            avg_volume = float(volume.tail(min(lookback, len(volume))).mean()) if not volume.empty else None
            ret_5d = ((closes.iloc[-1] / closes.iloc[-5]) - 1) * 100 if len(closes) >= 5 else None
            ret_20d = ((closes.iloc[-1] / closes.iloc[-20]) - 1) * 100 if len(closes) >= 20 else None
            
            # Technical analysis if enabled
            if self.config.include_technical_analysis and len(closes) >= 20:
                hist = self.compute_technical_indicators(hist)
            
            return {
                "available": True,
                "last_close": round(last_close, 2),
                "period_return_pct": round(pct_change_total, 2),
                "return_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
                "return_20d_pct": round(ret_20d, 2) if ret_20d is not None else None,
                "avg_volume": int(avg_volume) if avg_volume else None,
                "observations": len(closes),
                "lookback_days": lookback,
                "dataframe": hist,  # Return the full DataFrame with technical indicators
                "raw_data": {
                    "close": closes.tolist(),
                    "volume": volume.tolist(),
                    "high": highs.tolist(),
                    "low": lows.tolist(),
                } if self.config.include_technical_analysis else None,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error(f"Market data fetch failed: {exc}")
            return {"available": False, "reason": f"exception: {exc}"}
    
    def compute_technical_indicators(self, df) -> Any:
        """Compute technical indicators and add them to the DataFrame."""
        try:
            self.logger.info("Computing technical indicators")
            close_list = df["Close"].tolist()
            high_list = df["High"].tolist()
            low_list = df["Low"].tolist()
            
            # Moving averages
            sma_20 = calculate_moving_average(close_list, 20)
            sma_50 = calculate_moving_average(close_list, 50) if len(close_list) >= 50 else None
            
            # RSI
            rsi = calculate_rsi(close_list, 14)
            
            # MACD
            macd_line, signal_line, histogram = calculate_macd(close_list)
            
            # Add indicators to DataFrame (pad with NaN if different lengths)
            if len(sma_20) == len(df):
                df["SMA_20"] = sma_20
            if sma_50 and len(sma_50) == len(df):
                df["SMA_50"] = sma_50
            if len(rsi) == len(df):
                df["RSI"] = rsi
            if len(macd_line) == len(df):
                df["MACD"] = macd_line
                df["MACD_Signal"] = signal_line
                df["MACD_Histogram"] = histogram
            
            # Store summary for reporting
            current_price = close_list[-1]
            current_sma20 = sma_20[-1] if sma_20 and len(sma_20) > 0 else None
            current_sma50 = sma_50[-1] if sma_50 and len(sma_50) > 0 else None
            current_rsi = rsi[-1] if rsi and len(rsi) > 0 else None
            current_macd = macd_line[-1] if macd_line and len(macd_line) > 0 else None
            current_signal = signal_line[-1] if signal_line and len(signal_line) > 0 else None
            
            self.technical_indicators = {
                "sma_20": round(current_sma20, 2) if current_sma20 else None,
                "sma_50": round(current_sma50, 2) if current_sma50 else None,
                "rsi": round(current_rsi, 2) if current_rsi else None,
                "macd": round(current_macd, 4) if current_macd else None,
                "macd_signal": round(current_signal, 4) if current_signal else None,
                "price_vs_sma20": round((current_price / current_sma20 - 1) * 100, 2) if current_sma20 else None,
                "price_vs_sma50": round((current_price / current_sma50 - 1) * 100, 2) if current_sma50 else None,
                "macd_bullish": (current_macd > current_signal) if (current_macd and current_signal) else None,
                "rsi_oversold": current_rsi < 30 if current_rsi else None,
                "rsi_overbought": current_rsi > 70 if current_rsi else None,
            }
            
            return df
            
        except Exception as exc:
            self.logger.warning(f"Technical analysis failed: {exc}")
            self.technical_indicators = {"error": str(exc)}
            return df


__all__ = ["MarketDataProvider"]