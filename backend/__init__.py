"""
FlexLLama backend package.

This package provides functionality for managing and running llama.cpp models.
"""

from .config import ConfigManager
from .runner import RunnerManager, RunnerProcess

__version__ = "0.1.6"
__all__ = ["ConfigManager", "RunnerManager", "RunnerProcess"]
