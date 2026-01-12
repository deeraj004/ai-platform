"""
Agentic RAG Orchestrator Module
Provides the main orchestration components for the system
"""

try:
    from .prechecks import PreChecksEngine, PreChecksSummary, PreCheckResult
    PRECHECKS_AVAILABLE = True
except ImportError:
    PreChecksEngine = None
    PreChecksSummary = None
    PreCheckResult = None
    PRECHECKS_AVAILABLE = False

try:
    from .policy_router import policy_router, ExecutionPlan, ExecutionEdge
    POLICY_ROUTER_AVAILABLE = True
except ImportError:
    policy_router = None
    ExecutionPlan = None
    ExecutionEdge = None
    POLICY_ROUTER_AVAILABLE = False

from .agent_orchestrator import agent_orchestrator, AgentOrchestrator
from .state_schema import (
    AgenticState
)

__all__ = [
    'PreChecksEngine',
    'PreChecksSummary', 
    'PreCheckResult',
    'policy_router',
    'ExecutionPlan',
    'ExecutionEdge',
    'agent_orchestrator',
    'AgentOrchestrator',
    'AgenticState'
]
