"""
Observability utilities with LangSmith integration.
Falls back gracefully if LangSmith is not available or not configured.
"""
from typing import Optional, Dict, Any, Callable
from functools import wraps
from settings import logger, LANGSMITH_API_KEY, LANGSMITH_TRACING

# Try to import LangSmith
try:
    from langsmith import traceable, Client
    LANGSMITH_IMPORTABLE = True
except ImportError:
    LANGSMITH_IMPORTABLE = False
    # Create dummy decorator
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Check if LangSmith is properly configured
LANGSMITH_AVAILABLE = (
    LANGSMITH_IMPORTABLE and 
    LANGSMITH_API_KEY and 
    LANGSMITH_TRACING
)

if LANGSMITH_AVAILABLE:
    logger.info("LangSmith available and configured - observability enabled")
elif LANGSMITH_IMPORTABLE:
    if not LANGSMITH_API_KEY:
        logger.info("LangSmith installed but API key not configured - using basic logging")
    elif not LANGSMITH_TRACING:
        logger.info("LangSmith installed but tracing disabled - using basic logging")
    else:
        logger.info("LangSmith available but not properly configured - using basic logging")
else:
    logger.info("LangSmith not available - using basic logging for observability")


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
    Falls back to basic logging if LangSmith unavailable or not configured.
    
    Usage:
        @trace_agent("intent_classifier")
        def process(self, state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        if LANGSMITH_AVAILABLE:
            # Use LangSmith tracing with error handling
            try:
                # Wrap the function with traceable, but also add error handling
                @traceable(name=agent_name, **kwargs)
                @wraps(func)
                def langsmith_wrapper(*args, **kw):
                    logger.info(f"[LangSmith] Starting {agent_name}")
                    try:
                        result = func(*args, **kw)
                        logger.info(f"[LangSmith] Completed {agent_name}")
                        return result
                    except Exception as e:
                        logger.error(f"[LangSmith] Error in {agent_name}: {e}")
                        raise
                
                # Wrap again to catch LangSmith-specific errors (like 403, network issues)
                @wraps(func)
                def safe_wrapper(*args, **kw):
                    try:
                        return langsmith_wrapper(*args, **kw)
                    except Exception as e:
                        # Check if it's a LangSmith error (403, network, etc.)
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['langsmith', '403', 'forbidden', 'api.smith.langchain.com']):
                            # LangSmith error - fall back to basic logging for this call
                            logger.debug(f"LangSmith error for {agent_name}, using basic logging: {e}")
                            import time
                            start_time = time.time()
                            try:
                                result = func(*args, **kw)
                                elapsed = time.time() - start_time
                                logger.info(f"[Observability] Completed {agent_name} in {elapsed:.2f}s")
                                return result
                            except Exception as func_error:
                                elapsed = time.time() - start_time
                                logger.error(f"[Observability] Error in {agent_name} after {elapsed:.2f}s: {func_error}")
                                raise
                        else:
                            # Re-raise non-LangSmith errors
                            raise
                
                return safe_wrapper
            except Exception as e:
                # If LangSmith fails to initialize, fall back to basic logging
                logger.warning(f"LangSmith tracing failed for {agent_name}, falling back to basic logging: {e}")
                # Fall through to basic logging
        
        # Basic logging fallback (used if LangSmith unavailable or failed)
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
    """Log agent metrics (to LangSmith if available and configured, otherwise to logger)."""
    if LANGSMITH_AVAILABLE:
        client = get_langsmith_client()
        if client:
            try:
                # Log to LangSmith
                # Note: This is a simplified version - adjust based on your LangSmith setup
                logger.debug(f"[LangSmith] Metrics for {agent_name}: {metrics}")
            except Exception as e:
                # Silently fail - don't spam logs if LangSmith is not accessible
                logger.debug(f"Failed to log metrics to LangSmith (this is expected if LangSmith is not configured): {e}")
    
    # Always log to standard logger
    logger.info(f"[Metrics] {agent_name}: {metrics}")

