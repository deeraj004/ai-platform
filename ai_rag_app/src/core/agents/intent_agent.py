"""
Intent & Validation Agent
ReAct Agent with NO tools - understands user intent, handles greetings, normalizes queries.
"""
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import json
import re
import threading

from src.core.orchestrator.state_schema import AgenticState
from src.core.utils.observability import trace_agent
from settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_MODEL,
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
    logger
)


class IntentClassifierAgent:
    """
    Intent & Validation Agent - ReAct Agent with NO tools.
    
    Purpose:
    - Understand user intent
    - Handle greetings
    - Normalize spelling
    - Decide if early exit is possible
    - Extract domains/entities from user query
    
    Singleton pattern for efficient resource reuse.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(IntentClassifierAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize only once (singleton pattern)."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
        self.config = config or {}
        self.llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            model=AZURE_OPENAI_MODEL,
            temperature=0.3  # Lower temperature for more deterministic intent classification
        )
        
        self.system_prompt = INTENT_CLASSIFIER_SYSTEM_PROMPT
        IntentClassifierAgent._initialized = True
    
    @trace_agent("intent_classifier")
    def process(self, state: AgenticState) -> AgenticState:
        """
        Process the user query and classify intent.
        
        Args:
            state: AgenticState with user_query and conversation_history
            
        Returns:
            Updated AgenticState with intent classification results
        """
        try:
            logger.info(f"Intent Classifier: Processing query - {state.user_query[:100]}...")
            
            # Build conversation context
            conversation_context = ""
            if state.conversation_history:
                recent_history = state.conversation_history[-2:]  # Last 2 turns
                conversation_context = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('message', '')}"
                    for msg in recent_history
                ])
            
            # Create prompt
            user_prompt = f"""User Query: {state.user_query}

Conversation History:
{conversation_context if conversation_context else "No previous conversation"}

Analyze this query and provide your classification in the JSON format specified."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get LLM response
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse entire response
                intent_data = json.loads(response_text)
            
            # Update state with intent classification results
            state.normalized_query = intent_data.get("normalized_query", state.user_query)
            state.exit_early = intent_data.get("early_exit", False)
            
            # Store intent data in metadata
            state.metadata["intent_classification"] = {
                "user_intent": intent_data.get("user_intent", "query"),
                "is_greetings": intent_data.get("is_greetings", False),
                "is_RAG_required": intent_data.get("is_RAG_required", True),
                "domains_or_entities": intent_data.get("domains_or_entities_from_user_query", []),
                "confidence": intent_data.get("confidence", 0.8)
            }
            
            # If early exit (greeting or invalid), set answer
            if state.exit_early:
                state.answer = intent_data.get("answer", "Hello! I'm your banking assistant. How can I help you with your banking needs today?")
                state.orchestration_status = state.orchestration_status.__class__.EARLY_EXIT
                logger.info("Intent Classifier: Early exit triggered")
            
            logger.info(f"Intent Classifier: Intent={intent_data.get('user_intent')}, "
                       f"RAG_required={intent_data.get('is_RAG_required')}, "
                       f"Domains={intent_data.get('domains_or_entities_from_user_query')}")
            
            return state
            
        except Exception as e:
            logger.error(f"Intent Classifier Agent error: {e}", exc_info=True)
            # Fallback: continue with default values
            state.metadata["intent_classification"] = {
                "user_intent": "query",
                "is_greetings": False,
                "is_RAG_required": True,
                "domains_or_entities": [],
                "error": str(e)
            }
            # Default to banking domain if extraction fails
            if not state.metadata["intent_classification"].get("domains_or_entities"):
                state.metadata["intent_classification"]["domains_or_entities"] = ["accounts"]
            return state


def intent_classifier_node(state: AgenticState) -> AgenticState:
    """
    LangGraph node function for Intent Classifier Agent.
    
    Args:
        state: AgenticState
        
    Returns:
        Updated AgenticState
    """
    agent = IntentClassifierAgent()
    return agent.process(state)

