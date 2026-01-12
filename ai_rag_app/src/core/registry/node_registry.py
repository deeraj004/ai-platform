"""
Node Registry for Banking Agentic RAG System
Manages all available agents, tools, and nodes that can be used in banking workflows.
Uses LangGraph native framework agents.
"""
from typing import Dict, Type, Any, Optional, Callable
from settings import logger, FRAMEWORK_LANGGRAPH_NATIVE

# Import new banking-focused agents
from src.core.agents.intent_agent import IntentClassifierAgent
from src.core.agents.planning_agent import PlanningAgent
from src.core.agents.execution_agent import ExecutionAgent
from src.core.agents.critic_evaluator_agent import CriticEvaluatorAgent

class NodeRegistry:
    """
    Central registry for all available nodes (agents, tools, evaluators)
    that can be used in workflows
    """
    
    def __init__(self):
        self._nodes: Dict[str, Type] = {}
        self._node_metadata: Dict[str, Dict[str, Any]] = {}
        self._framework_nodes: Dict[str, Dict[str, Any]] = {}  # NEW: Track framework-specific nodes
        self._register_default_nodes()
    
    def _register_default_nodes(self):
        """Register all default nodes for banking agentic system"""
        
        # Banking Agentic Agents (LangGraph Native)
        self.register_node("intent_classifier", IntentClassifierAgent, {
            "type": "agent",
            "framework": FRAMEWORK_LANGGRAPH_NATIVE,
            "category": "banking_agents",
            "description": "Identifies user intent and banking domain category",
            "required_inputs": ["user_query", "conversation_history"],
            "outputs": ["intent", "domain", "normalized_query"],
            "execution_conditions": ["always"]
        })
        
        self.register_node("planning", PlanningAgent, {
            "type": "agent",
            "framework": FRAMEWORK_LANGGRAPH_NATIVE,
            "category": "banking_agents",
            "description": "Creates execution plan for banking APIs",
            "required_inputs": ["user_query", "domains"],
            "outputs": ["execution_plan"],
            "execution_conditions": ["always"]
        })
        
        self.register_node("execution", ExecutionAgent, {
            "type": "agent",
            "framework": FRAMEWORK_LANGGRAPH_NATIVE,
            "category": "banking_agents",
            "description": "Executes banking APIs from execution plan",
            "required_inputs": ["execution_plan", "tool_registry"],
            "outputs": ["tool_results"],
            "execution_conditions": ["always"]
        })
        
        self.register_node("evaluator", CriticEvaluatorAgent, {
            "type": "agent",
            "framework": FRAMEWORK_LANGGRAPH_NATIVE,
            "category": "banking_agents",
            "description": "Evaluates banking API execution quality and compliance",
            "required_inputs": ["tool_results", "execution_plan"],
            "outputs": ["evaluation_scores", "feedback"],
            "execution_conditions": ["always"]
        })
        
        
        logger.info(f"Registered {len(self._nodes)} default nodes for banking agentic system")
    
    
    def register_node(self, node_name: str, node_class: Type, metadata: Dict[str, Any] = None):
        """
        Register a new node in the registry
        
        Args:
            node_name: Unique identifier for the node
            node_class: Class that implements the node functionality
            metadata: Additional information about the node
        """
        if node_name in self._nodes:
            logger.warning(f"Node {node_name} already registered, overwriting")
        
        self._nodes[node_name] = node_class
        self._node_metadata[node_name] = metadata or {}
        logger.info(f"Registered node: {node_name}")
    
    def get_node(self, node_name: str) -> Optional[Type]:
        """Get a node class by name"""
        return self._nodes.get(node_name)
    
    def get_node_metadata(self, node_name: str) -> Dict[str, Any]:
        """Get metadata for a specific node"""
        return self._node_metadata.get(node_name, {})
    
    def list_nodes_by_category(self, category: str) -> Dict[str, Type]:
        """List all nodes in a specific category"""
        return {
            name: node_class for name, node_class in self._nodes.items()
            if self._node_metadata.get(name, {}).get("category") == category
        }
    
    def list_all_nodes(self) -> Dict[str, Type]:
        """List all registered nodes"""
        return self._nodes.copy()

    
    def is_node_available(self, node_name: str) -> bool:
        """Check if a node is available (registered and has a class)"""
        return node_name in self._nodes and self._nodes[node_name] is not None
    
    def get_node_for_framework(
        self,
        node_name: str,
        framework: str = None,
        config: Dict[str, Any] = None
    ) -> Optional[Any]:
        """
        Get node implementation for specified framework.
        
        Args:
            node_name: Node name (e.g., 'rag', 'evaluator')
            framework: Framework to use (FRAMEWORK_LANGGRAPH_NATIVE)
            config: Configuration dict for the node
            
        Returns:
            Node class (for LangGraph)
        """
        if framework is None:
            framework = FRAMEWORK_LANGGRAPH_NATIVE
            
        if framework == FRAMEWORK_LANGGRAPH_NATIVE:
            # Return existing LangGraph agent class
            return self.get_node(node_name)
        else:
            logger.warning(f"Unknown framework '{framework}', using LangGraph native")
            return self.get_node(node_name)
    
    def supports_framework(self, node_name: str, framework: str) -> bool:
        """
        Check if a node supports the specified framework.
        
        Args:
            node_name: Node name
            framework: Framework name (FRAMEWORK_LANGGRAPH_NATIVE)
            
        Returns:
            True if node supports the framework
        """
        if framework == FRAMEWORK_LANGGRAPH_NATIVE:
            return self.is_node_available(node_name)
        return False

# Global node registry instance
node_registry = NodeRegistry()
