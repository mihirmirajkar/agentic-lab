"""Web search functionality for ticker research."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import time
import logging

try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional
    DDGS = None  # type: ignore

from agentic_lab.core.exceptions import AgentError
from .config import TickerResearchConfig


class SearchProvider:
    """Handles web search functionality for ticker research."""
    
    def __init__(self, config: TickerResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._search_backend = DDGS if DDGS else None
    
    def timelimit_token(self) -> str:
        """Convert lookback days to search timelimit token."""
        days = self.config.lookback_days or 21
        if days <= 7:
            return "w"
        if days <= 30:
            return "m"
        if days <= 365:
            return "y"
        return "y"
    
    def date_range_label(self) -> str:
        """Generate human-readable date range label."""
        return f"last {self.config.lookback_days} days"
    
    def build_queries(self) -> List[Tuple[str, str]]:
        """Build categorized search queries based on config."""
        base = self.config.ticker.upper()
        horizon = self.date_range_label()
        queries: List[Tuple[str, str]] = [
            ("news", f"{base} stock news {horizon}"),
            ("catalysts", f"{base} guidance update {horizon}"),
            ("sentiment", f"{base} analyst rating sentiment {horizon}"),
            ("macro", f"{base} sector impact macro {horizon}"),
        ]
        
        if self.config.include_financial_terms:
            queries.extend([
                ("fundamentals", f"{base} earnings results {horizon}"),
                ("fundamentals", f"{base} revenue growth {horizon}"),
                ("risk", f"{base} litigation risk {horizon}"),
            ])
        
        # Deduplicate by text while keeping first category
        seen = set()
        dedup: List[Tuple[str, str]] = []
        for cat, q in queries:
            if q not in seen:
                seen.add(q)
                dedup.append((cat, q))
        return dedup
    
    def search(self, queries: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Execute web search across query categories."""
        if not self._search_backend:
            raise AgentError("ddgs not installed. Install with agentic-lab[research]")
        
        ddgs = self._search_backend()
        results: List[Dict[str, Any]] = []
        seen_urls = set()
        timelimit = self.timelimit_token()
        
        self.logger.info(f"Starting search across {len(queries)} query variants (timelimit token '{timelimit}')")
        
        for category, q in queries:
            self.logger.debug(f"Query '{q}' (category={category})")
            try:
                for item in ddgs.text(q, region="wt-wt", safesearch="moderate", timelimit=timelimit):
                    if len(results) >= self.config.max_results:
                        break
                    url = item.get("href") or item.get("url")
                    if url and url in seen_urls:
                        continue
                    seen_urls.add(url)
                    results.append({
                        "query": q,
                        "category": category,
                        "title": item.get("title"),
                        "snippet": item.get("body") or item.get("description"),
                        "url": url,
                    })
                if len(results) >= self.config.max_results:
                    break
            finally:  # polite pacing between queries
                time.sleep(0.25)
        
        self.logger.info(f"Search complete – gathered {len(results)} unique sources")
        return results


__all__ = ["SearchProvider"]