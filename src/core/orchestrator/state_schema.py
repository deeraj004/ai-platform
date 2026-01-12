from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


@dataclass
class CriticFeedback:
    """Critic agent evaluation feedback (dataclass for compatibility)"""
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    bias_detection_score: float = 0.0
    logical_consistency_score: float = 0.0
    overall_quality_score: float = 0.0
    feedback_summary: str = ""
    detailed_feedback: List[Dict[str, str]] = field(default_factory=list)


class OrchestrationStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EARLY_EXIT = "EARLY_EXIT"


class AgenticState(BaseModel):
    """
    AgenticState: canonical runtime schema for the LangGraph orchestrator.
    Tracks user context, orchestration flow, and intermediate results across agent nodes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---- Core context ----
    state: Optional[Dict[str, Any]] = None
    user_info: Dict[str, Any] = Field(default_factory=dict)

    session_id: str
    conversation_session_id: str
    request_id: str
    
    user_query: str
    normalized_query: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    open_search_alias_name: Optional[str] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    # ---- Orchestration / Graph runtime ---
    current_node_id: Optional[str] = None
    execution_plan: Optional[Any] = None
    evaluation_scores: Dict[str, float] = Field(default_factory=dict)
    tool_results: Optional[Dict[str, Any]] = Field(default_factory=dict)
    reranked_results: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None
    orchestration_status: OrchestrationStatus = OrchestrationStatus.IN_PROGRESS
    response: Optional[Dict[str, Any]] = None
    exit_early: bool = False
    critic_feedback: Optional[CriticFeedback] = None
    # ---- RAG specific fields ----
    embedding_query: Optional[Any] = None
    chunk_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    augmented_user_query: Optional[List[str]] = None
    sources: Optional[List[str]] = Field(default_factory=list)  # Source documents for citations
    document_metadata: Optional[List[Dict[str, Any]]] = Field(default_factory=list)  # Document metadata for citations


    @property
    def query(self):
        """Alias for user_query to maintain backwards compatibility."""
        return self.user_query
    
    @query.setter
    def query(self, value):
        """Setter for query property."""
        self.user_query = value

    @property
    def alias(self):
        """Alias for open_search_alias_name to maintain backwards compatibility."""
        return self.open_search_alias_name
    
    @alias.setter
    def alias(self, value):
        """Setter for alias property."""
        self.open_search_alias_name = value

    # Add dict-like behavior for legacy compatibility
    def get(self, key: str, default=None):
        """Get attribute with default value, dict-like behavior."""
        return getattr(self, key, default)
    
    def update(self, *args, **kwargs):
        """
        Update attributes from kwargs or dict, dict-like behavior.
        Supports both state.update(key=value) and state.update({'key': 'value'})
        """
        # Handle positional dict argument
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                raise TypeError(f"update() takes 1 positional argument but {len(args)} were given")
        
        # Update attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in ['vector_search_results', 'tool_results']:
                # Store unknown keys in tool_results
                if not self.tool_results:
                    self.tool_results = {}
                self.tool_results[key] = value
    
    def __getitem__(self, key):
        """Dictionary-style access."""
        if hasattr(self, key):
            return getattr(self, key)
        if self.tool_results and key in self.tool_results:
            return self.tool_results[key]
        raise KeyError(f"Key '{key}' not found in AgenticState")
    
    def __setitem__(self, key, value):
        """Dictionary-style assignment."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            if not self.tool_results:
                self.tool_results = {}
            self.tool_results[key] = value

    def __contains__(self, key):
        """Check if key exists in state."""
        return hasattr(self, key) or (self.tool_results and key in self.tool_results)
