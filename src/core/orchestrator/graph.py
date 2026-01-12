"""
LangGraph Workflow
Multi-agent orchestration graph with state management.
"""
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.core.orchestrator.state_schema import AgenticState, OrchestrationStatus
from src.core.agents.intent_agent import intent_classifier_node
from src.core.agents.tool_router_agent import tool_router_node
from src.core.agents.planning_agent import planning_node
from src.core.agents.execution_agent import execution_node
from src.core.agents.critic_evaluator_agent import critic_evaluator_node
from src.core.registry.tool_registry import tool_registry
from src.core.registry.node_registry import node_registry
from settings import logger


def should_continue(state: AgenticState) -> Literal["planning", "end"]:
    """
    Conditional edge: Check if we should continue or exit early.
    
    Args:
        state: AgenticState
        
    Returns:
        Next node name or "end"
    """
    if state.exit_early:
        logger.info("Graph: Early exit condition met")
        return "end"
    
    if state.orchestration_status == OrchestrationStatus.EARLY_EXIT:
        logger.info("Graph: Early exit status detected")
        return "end"
    
    return "planning"


def should_execute(state: AgenticState) -> Literal["execution", "end"]:
    """
    Conditional edge: Check if we should execute tools or end.
    
    Args:
        state: AgenticState
        
    Returns:
        Next node name or "end"
    """
    if not state.execution_plan:
        logger.warning("Graph: No execution plan, ending")
        return "end"
    
    steps = state.execution_plan.get("steps", [])
    if not steps:
        logger.warning("Graph: No steps in execution plan, ending")
        return "end"
    
    return "execution"


def should_evaluate(state: AgenticState) -> Literal["response_generator", "end"]:
    """
    Conditional edge: Check if we should generate response or end.
    
    Args:
        state: AgenticState
        
    Returns:
        Next node name or "end"
    """
    if not state.tool_results:
        logger.warning("Graph: No tool results to process, ending")
        return "end"
    
    return "response_generator"


def should_retry(state: AgenticState) -> Literal["planning", "end"]:
    """
    Conditional edge: Check if we should retry based on evaluation.
    
    Args:
        state: AgenticState
        
    Returns:
        Next node name or "end"
    """
    evaluation = state.metadata.get("evaluation", {})
    should_retry_flag = evaluation.get("should_retry", False)
    
    if should_retry_flag:
        logger.info("Graph: Retry recommended by evaluator")
        return "planning"
    
    return "end"


def create_agentic_graph() -> StateGraph:
    """
    Create the LangGraph workflow for multi-agent orchestration.
    
    Returns:
        Compiled StateGraph
    """
    logger.info("Creating agentic orchestration graph")
    
    # Create state graph
    workflow = StateGraph(AgenticState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    
    # Add Tool Router node
    def tool_router_wrapper(state: AgenticState) -> AgenticState:
        """Wrapper to provide tool registry to tool router"""
        from settings import TOOL_ROUTER_TOP_K, TOOL_ROUTER_MAX_CACHE_SIZE
        from src.core.agents.tool_router_agent import ToolRouterAgent
        
        tool_registry_dict = {
            domain: tool_registry.get_tools_for_domain(domain)
            for domain in tool_registry.list_domains()
        }
        
        # Create agent with config
        agent = ToolRouterAgent(config={
            "top_k": TOOL_ROUTER_TOP_K,
            "max_cache_size": TOOL_ROUTER_MAX_CACHE_SIZE
        })
        return agent.process(state, tool_registry_dict)
    
    workflow.add_node("tool_router", tool_router_wrapper)
    
    def planning_wrapper(state: AgenticState) -> AgenticState:
        """Wrapper to provide tool registry to planning node"""
        tool_registry_dict = tool_registry.list_tools_by_domain()
        return planning_node(state, tool_registry_dict)
    
    def execution_wrapper(state: AgenticState) -> AgenticState:
        """Wrapper to provide tool registry to execution node"""
        tool_registry_dict = {
            domain: tool_registry.get_tools_for_domain(domain)
            for domain in tool_registry.list_domains()
        }
        return execution_node(state, tool_registry_dict, node_registry)
    
    # Add response generation node
    def response_generator_node(state: AgenticState) -> AgenticState:
        """Generate final response from tool results"""
        from src.core.utils.azure_chatopenai import azure_chatopenai
        
        try:
            logger.info("Response Generator: Generating final answer from tool results")
            
            # Extract results from tool execution
            tool_results = state.tool_results or {}
            context_parts = []
            
            for step_key, step_result in tool_results.items():
                if isinstance(step_result, dict) and step_result.get("status") == "success":
                    result = step_result.get("result", "")
                    if result:
                        context_parts.append(str(result))
            
            context = "\n\n".join(context_parts) if context_parts else "No results available"
            
            # Generate response using Azure OpenAI
            if state.tool_results and any(r.get("status") == "success" for r in tool_results.values()):
                # Use the generate_response method
                ranked_results = (state.sources or [], state.chunk_data.get("chunk_data", []) if isinstance(state.chunk_data, dict) else [])
                
                answer = azure_chatopenai.generate_response(
                    query=state.normalized_query or state.user_query,
                    context=context,
                    ranked_search_results=ranked_results
                )
                
                state.answer = answer if isinstance(answer, str) else str(answer)
            else:
                state.answer = "I couldn't retrieve the information you requested. Please try rephrasing your question."
            
            logger.info("Response Generator: Final answer generated")
            return state
            
        except Exception as e:
            logger.error(f"Response Generator error: {e}", exc_info=True)
            state.answer = f"An error occurred while generating the response: {str(e)}"
            return state
    
    workflow.add_node("planning", planning_wrapper)
    workflow.add_node("execution", execution_wrapper)
    workflow.add_node("response_generator", response_generator_node)
    workflow.add_node("evaluation", critic_evaluator_node)
    
    # Set entry point
    workflow.set_entry_point("intent_classifier")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "intent_classifier",
        should_continue,
        {
            "planning": "tool_router",  # Route to tool_router first
            "end": END
        }
    )
    
    # Add edge from tool_router to planning
    workflow.add_edge("tool_router", "planning")
    
    workflow.add_conditional_edges(
        "planning",
        should_execute,
        {
            "execution": "execution",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "execution",
        should_evaluate,
        {
            "response_generator": "response_generator",
            "end": END
        }
    )
    
    workflow.add_edge("response_generator", "evaluation")
    
    # Add a retry handler node that resets state before replanning
    def retry_handler(state: AgenticState) -> AgenticState:
        """Reset state for retry"""
        logger.info("Graph: Resetting state for retry")
        state.execution_plan = None
        state.tool_results = {}
        state.metadata["retry_count"] = state.metadata.get("retry_count", 0) + 1
        return state
    
    workflow.add_node("retry_handler", retry_handler)
    
    workflow.add_conditional_edges(
        "evaluation",
        should_retry,
        {
            "planning": "retry_handler",
            "end": END
        }
    )
    
    # Connect retry handler to tool_router (so tools are re-routed on retry)
    workflow.add_edge("retry_handler", "tool_router")
    
    # Compile graph with memory
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    
    logger.info("Agentic orchestration graph created successfully")
    return compiled_graph


# Global graph instance 
_agentic_graph = None

def get_agentic_graph():
    """Get or create the agentic graph (lazy initialization)"""
    global _agentic_graph
    if _agentic_graph is None:
        _agentic_graph = create_agentic_graph()
    return _agentic_graph

