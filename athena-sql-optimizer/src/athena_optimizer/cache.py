"""Simple in-memory cache with TTL support."""

import time
from typing import Any, Optional, Callable
from functools import wraps
from threading import Lock


class TTLCache:
    """Thread-safe in-memory cache with time-to-live expiration."""

    def __init__(self, default_ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl_seconds: Default time-to-live in seconds (default: 5 minutes)
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if time.time() > expiry:
                # Expired, remove it
                del self._cache[key]
                return None

            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl

        with self._lock:
            self._cache[key] = (value, expiry)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def clear_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time > expiry
            ]
            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of entries in cache
        """
        with self._lock:
            return len(self._cache)


def cached(ttl_seconds: Optional[int] = None, cache_instance: Optional[TTLCache] = None):
    """
    Decorator to cache function results with TTL.

    Args:
        ttl_seconds: Time-to-live in seconds
        cache_instance: Cache instance to use (creates new if None)

    Usage:
        @cached(ttl_seconds=300)
        def expensive_function(arg1, arg2):
            return result
    """
    if cache_instance is None:
        cache_instance = TTLCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Try to get from cache
            cached_value = cache_instance.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl_seconds)

            return result

        # Expose cache management methods
        wrapper.cache = cache_instance
        wrapper.clear_cache = cache_instance.clear

        return wrapper

    return decorator
