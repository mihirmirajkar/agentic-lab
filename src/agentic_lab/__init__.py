"""
Agentic Lab - A playground for building, experimenting, and scaling agentic systems.

This package provides tools and frameworks for developing autonomous agents,
starting with trading agents and expanding to multi-domain workflows.
"""

__version__ = "0.1.0"
__author__ = "Mihir Mirajkar"
__description__ = (
    "A playground for building, experimenting, and scaling agentic systems"
)

# Core modules - import the main components
from .core import *  # noqa: F401,F403

__all__ = [
    "__version__",
    "__author__",
    "__description__",
]
