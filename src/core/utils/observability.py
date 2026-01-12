"""
Observability utilities with LangSmith integration.
Falls back gracefully if LangSmith is not available.
"""
from typing import Optional, Dict, Any, Callable
from functools import wraps
from settings import logger

# Try to import LangSmith
try:
    from langsmith import traceable, Client
    LANGSMITH_AVAILABLE = True
    logger.info("LangSmith available - observability enabled")
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.info("LangSmith not available - using basic logging for observability")
    # Create dummy decorator
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def get_langsmith_client() -> Optional[Any]:
    """Get LangSmith client if available."""
    if not LANGSMITH_AVAILABLE:
        return None
    try:
        return Client()
    except Exception as e:
        logger.warning(f"Could not initialize LangSmith client: {e}")
        return None


def trace_agent(agent_name: str, **kwargs):
    """
    Decorator to trace agent execution with LangSmith.
    Falls back to basic logging if LangSmith unavailable.
    
    Usage:
        @trace_agent("intent_classifier")
        def process(self, state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        if LANGSMITH_AVAILABLE:
            # Use LangSmith tracing
            @traceable(name=agent_name, **kwargs)
            @wraps(func)
            def wrapper(*args, **kw):
                logger.info(f"[LangSmith] Starting {agent_name}")
                try:
                    result = func(*args, **kw)
                    logger.info(f"[LangSmith] Completed {agent_name}")
                    return result
                except Exception as e:
                    logger.error(f"[LangSmith] Error in {agent_name}: {e}")
                    raise
            return wrapper
        else:
            # Fallback to basic logging
            @wraps(func)
            def wrapper(*args, **kw):
                logger.info(f"[Observability] Starting {agent_name}")
                import time
                start_time = time.time()
                try:
                    result = func(*args, **kw)
                    elapsed = time.time() - start_time
                    logger.info(f"[Observability] Completed {agent_name} in {elapsed:.2f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(f"[Observability] Error in {agent_name} after {elapsed:.2f}s: {e}")
                    raise
            return wrapper
    return decorator


def log_agent_metrics(agent_name: str, metrics: Dict[str, Any]):
    """Log agent metrics (to LangSmith if available, otherwise to logger)."""
    if LANGSMITH_AVAILABLE:
        client = get_langsmith_client()
        if client:
            try:
                # Log to LangSmith
                # Note: This is a simplified version - adjust based on your LangSmith setup
                logger.debug(f"[LangSmith] Metrics for {agent_name}: {metrics}")
            except Exception as e:
                logger.warning(f"Failed to log metrics to LangSmith: {e}")
    
    # Always log to standard logger
    logger.info(f"[Metrics] {agent_name}: {metrics}")

