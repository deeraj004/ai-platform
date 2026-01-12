"""
Planning Agent
Translates user intent into an execution plan with tool sequences.
"""
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import threading

from src.core.orchestrator.state_schema import AgenticState
from src.core.registry.tool_registry import tool_registry as full_tool_registry
from src.core.utils.observability import trace_agent
from settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_MODEL,
    PLANNING_AGENT_SYSTEM_PROMPT,
    logger
)


class PlanningAgent:
    """
    Planning Agent - Creates execution plans from user intent.
    
    Purpose:
    - Translate user intent into tool execution sequence
    - Build ordered workflow for complex queries
    - Map domains/entities to appropriate tools
    
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
                    cls._instance = super(PlanningAgent, cls).__new__(cls)
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
            temperature=0.5
        )
        
        self.system_prompt = PLANNING_AGENT_SYSTEM_PROMPT
        PlanningAgent._initialized = True
    
    @trace_agent("planning_agent")
    def process(self, state: AgenticState, tool_registry: Dict[str, List[str]]) -> AgenticState:
        """
        Create execution plan from user intent.
        """
        try:
            logger.info("Planning Agent: Creating execution plan")
            
            # Get intent classification from metadata
            intent_data = state.metadata.get("intent_classification", {})
            domains = intent_data.get("domains_or_entities", [])
            is_rag_required = intent_data.get("is_RAG_required", True)
            
            # Get filtered tools from tool router (if available)
            filtered_tool_names = state.metadata.get("filtered_tool_names", [])
            routed_tools_info = state.metadata.get("routed_tools", {})
            
            if filtered_tool_names and routed_tools_info:
                # Use filtered tools from router - build context from filtered tool names
                # We need to get descriptions from the full registry
                tool_descriptions = []
                for tool_name in filtered_tool_names:
                    metadata = full_tool_registry.get_tool_metadata(tool_name)
                    if metadata:
                        desc = metadata.get("description", "No description")
                        domain = metadata.get("domain", "unknown")
                        tool_descriptions.append(f"- {tool_name} ({domain}): {desc}")
                    else:
                        tool_descriptions.append(f"- {tool_name}: (No description available)")
                
                registry_context = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
                logger.info(
                    f"Planning Agent: Using {len(filtered_tool_names)} filtered tools from router "
                    f"(method: {routed_tools_info.get('routing_method', 'unknown')})"
                )
            else:
                # Fallback: use all tools (legacy behavior)
                registry_context = "\n".join([
                    f"Domain '{domain}': {', '.join(tools)}"
                    for domain, tools in tool_registry.items()
                ]) if tool_registry else "No tools available"
                logger.warning(
                    f"Planning Agent: No filtered tools available, using all {sum(len(tools) for tools in tool_registry.values())} tools "
                    f"(may be slow with many tools)"
                )
            
            # Create prompt
            user_prompt = f"""User Query: {state.normalized_query or state.user_query}

Identified Domains/Entities: {domains}

Available Tool Registry:
{registry_context}

RAG Required: {is_rag_required}

Create an execution plan to fulfill this query. Consider:
1. What tools are needed based on the domains
2. What order should tools be executed
3. What data needs to be passed between steps
4. Whether RAG/documentation search is needed

Provide your plan in the JSON format specified."""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get LLM response
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                execution_plan = json.loads(json_match.group())
            else:
                execution_plan = json.loads(response_text)
            
            # Store execution plan in state
            state.execution_plan = execution_plan
            state.metadata["planning"] = {
                "goal": execution_plan.get("goal", ""),
                "estimated_steps": execution_plan.get("estimated_steps", len(execution_plan.get("steps", []))),
                "steps_count": len(execution_plan.get("steps", []))
            }
            
            logger.info(f"Planning Agent: Created plan with {len(execution_plan.get('steps', []))} steps")
            logger.debug(f"Planning Agent: Goal - {execution_plan.get('goal', 'N/A')}")
            
            return state
            
        except Exception as e:
            logger.error(f"Planning Agent error: {e}", exc_info=True)
            # Fallback: create minimal plan
            state.execution_plan = {
                "goal": "Process user query",
                "steps": [
                    {
                        "step_id": 1,
                        "tool": "rag.retrieve_documents",
                        "description": "Retrieve relevant documents",
                        "inputs": {"query": state.user_query},
                        "depends_on": []
                    }
                ],
                "estimated_steps": 1
            }
            state.metadata["planning"] = {"error": str(e)}
            return state


def planning_node(state: AgenticState, tool_registry: Dict[str, List[str]]) -> AgenticState:
    """
    LangGraph node function for Planning Agent.
    
    Args:
        state: AgenticState
        tool_registry: Tool registry dict
        
    Returns:
        Updated AgenticState
    """
    agent = PlanningAgent()
    return agent.process(state, tool_registry)

