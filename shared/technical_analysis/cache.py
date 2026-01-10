# shared/technical_analysis/cache.py
"""
Caching functionality for technical analysis calculations.
"""

import hashlib
import json
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from ..constants import CacheConfig


class TechnicalAnalysisCache:
    """LRU cache for technical analysis calculations."""

    def __init__(self, cache_size: int = CacheConfig.DEFAULT_CACHE_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._config_hash: Optional[str] = None

    def get_config_hash(self, config: Dict) -> str:
        """Generate a deterministic hash for a configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_cached_data(
        self,
        ticker: str,
        date_range: Tuple[datetime, datetime],
        config_hash: str
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available."""
        cache_key = (ticker, date_range, config_hash)
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        return None

    def set_cached_data(
        self,
        ticker: str,
        date_range: Tuple[datetime, datetime],
        config_hash: str,
        data: pd.DataFrame
    ) -> None:
        """Store data in cache with LRU eviction."""
        cache_key = (ticker, date_range, config_hash)

        # If key exists, move to end
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self._cache[cache_key] = data
            return

        # Evict oldest if at capacity
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

        self._cache[cache_key] = data

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._config_hash = None

    @property
    def config_hash(self) -> Optional[str]:
        """Get the current configuration hash."""
        return self._config_hash

    @config_hash.setter
    def config_hash(self, value: str) -> None:
        """Set the configuration hash."""
        self._config_hash = value

    def __len__(self) -> int:
        """Return the number of cached items."""
        return len(self._cache)
