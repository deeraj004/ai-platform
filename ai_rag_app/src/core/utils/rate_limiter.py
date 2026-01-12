"""
Rate Limiting Utility
Implements token bucket algorithm for rate limiting.
Thread-safe and supports multiple rate limiters.
"""
import time
import threading
from typing import Dict, Optional
from collections import defaultdict
from settings import logger


class TokenBucket:
    """
    Token Bucket rate limiter implementation.
    Thread-safe and efficient.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens (burst capacity)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        
        Returns:
            True if tokens were acquired, False if rate limit exceeded
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            return self.tokens


class RateLimiter:
    """
    Rate limiter with support for multiple named limiters.
    Thread-safe and efficient.
    """
    
    def __init__(self):
        self._limiters: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
    
    def get_limiter(self, name: str, capacity: int, refill_rate: float) -> TokenBucket:
        """
        Get or create a rate limiter.
        
        Args:
            name: Name of the rate limiter
            capacity: Maximum tokens (burst capacity)
            refill_rate: Tokens per second
        
        Returns:
            TokenBucket instance
        """
        with self._lock:
            if name not in self._limiters:
                self._limiters[name] = TokenBucket(capacity, refill_rate)
            return self._limiters[name]
    
    def is_allowed(
        self,
        name: str,
        capacity: int,
        refill_rate: float,
        tokens: int = 1
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            name: Name of the rate limiter
            capacity: Maximum tokens (burst capacity)
            refill_rate: Tokens per second
            tokens: Number of tokens to consume (default: 1)
        
        Returns:
            True if allowed, False if rate limited
        """
        limiter = self.get_limiter(name, capacity, refill_rate)
        return limiter.acquire(tokens)
    
    def get_remaining(self, name: str) -> Optional[float]:
        """
        Get remaining tokens for a limiter.
        
        Args:
            name: Name of the rate limiter
        
        Returns:
            Number of remaining tokens or None if limiter doesn't exist
        """
        with self._lock:
            if name in self._limiters:
                return self._limiters[name].get_available_tokens()
            return None


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit(
    name: str,
    capacity: int,
    refill_rate: float,
    tokens: int = 1,
    error_message: Optional[str] = None
):
    """
    Decorator for rate limiting functions.
    
    Args:
        name: Name of the rate limiter
        capacity: Maximum tokens (burst capacity)
        refill_rate: Tokens per second
        tokens: Number of tokens to consume (default: 1)
        error_message: Custom error message if rate limited
    
    Usage:
        @rate_limit("api_calls", capacity=100, refill_rate=10.0)
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not _rate_limiter.is_allowed(name, capacity, refill_rate, tokens):
                error_msg = error_message or f"Rate limit exceeded for {name}"
                logger.warning(f"Rate limit exceeded: {name}")
                return {
                    "error": error_msg,
                    "rate_limited": True,
                    "limiter": name
                }
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_rate_limit(
    name: str,
    capacity: int,
    refill_rate: float,
    tokens: int = 1
) -> bool:
    """
    Check if operation is allowed by rate limiter.
    
    Args:
        name: Name of the rate limiter
        capacity: Maximum tokens (burst capacity)
        refill_rate: Tokens per second
        tokens: Number of tokens to consume (default: 1)
    
    Returns:
        True if allowed, False if rate limited
    """
    return _rate_limiter.is_allowed(name, capacity, refill_rate, tokens)

