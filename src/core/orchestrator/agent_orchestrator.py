"""
Agent Orchestrator
Main orchestrator that manages request lifecycle and coordinates the agentic workflow.
"""
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.orchestrator.state_schema import AgenticState, OrchestrationStatus
from settings import logger


class AgentOrchestrator:
    """
    Agent Orchestrator - Manages request lifecycle and coordinates workflow.
    
    Responsibilities:
    - Validate early_exit
    - Manage request lifecycle
    - Update state schema (request_id, user_id, session_state)
    - Handle exceptions, retries, fallbacks
    """
    
    def __init__(self):
        from src.core.orchestrator.graph import get_agentic_graph
        self.graph = get_agentic_graph()
        logger.info("Agent Orchestrator initialized")
    
    async def orchestrate_query(self, query_payload: Dict[str, Any]) -> str:
        """
        Orchestrate a user query through the agentic workflow.        
        """
        start_time = time.time()
        request_id = query_payload.get("request_id", str(uuid.uuid4()))
        
        try:
            logger.info(f"Orchestrator: Processing request {request_id}")
            
            # Create initial state
            state = AgenticState(
                state=None,
                user_info={
                    "user_id": query_payload.get("user_id", ""),
                    "session_id": query_payload.get("session_id", "")
                },
                session_id=query_payload.get("session_id", ""),
                conversation_session_id=query_payload.get("conversation_session_id", ""),
                request_id=request_id,
                user_query=query_payload.get("query", ""),
                normalized_query=None,
                conversation_history=query_payload.get("conversation_history", []),
                open_search_alias_name=query_payload.get("open_search_alias_name"),
                additional_params=query_payload.get("additional_params", {}),
                orchestration_status=OrchestrationStatus.IN_PROGRESS,
                exit_early=False
            )
            
            # Execute graph
            config = {"configurable": {"thread_id": state.conversation_session_id}}
            
            final_state = None
            async for state_update in self.graph.astream(state, config):
                # Process state updates
                for node_name, node_state in state_update.items():
                    logger.debug(f"Orchestrator: Node '{node_name}' completed")
                    final_state = node_state
            
            # If no final state from streaming, get final state
            if final_state is None:
                final_state = await self.graph.ainvoke(state, config)
            
            # Ensure final_state is an AgenticState object, not a dict
            if isinstance(final_state, dict):
                # Convert dict to AgenticState if needed
                try:
                    final_state = AgenticState(**final_state)
                except Exception as e:
                    logger.warning(f"Could not convert final_state dict to AgenticState: {e}, using original state")
                    final_state = state
            
            # Calculate processing time
            processing_time = time.time() - start_time
            if hasattr(final_state, 'processing_time'):
                final_state.processing_time = processing_time
            else:
                # If it's still a dict, update it
                if isinstance(final_state, dict):
                    final_state['processing_time'] = processing_time
                else:
                    # Store in metadata as fallback
                    if not hasattr(final_state, 'metadata'):
                        final_state.metadata = {}
                    final_state.metadata['processing_time'] = processing_time
            
            # Determine final status
            if hasattr(final_state, 'exit_early') and final_state.exit_early:
                final_state.orchestration_status = OrchestrationStatus.EARLY_EXIT
            elif hasattr(final_state, 'answer') and final_state.answer and not final_state.metadata.get("evaluation", {}).get("should_retry", False):
                final_state.orchestration_status = OrchestrationStatus.COMPLETED
            else:
                final_state.orchestration_status = OrchestrationStatus.COMPLETED
            
            # Build response
            response = self._build_response(final_state)
            
            logger.info(f"Orchestrator: Request {request_id} completed in {processing_time:.2f}s")
            logger.debug(f"Orchestrator: Final status = {final_state.orchestration_status}")
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator: Error processing request {request_id}: {e}", exc_info=True)
            return f"An error occurred while processing your request: {str(e)}"
    
    def _build_response(self, state: AgenticState) -> str:
        """
        Build final response from state.
        
        Args:
            state: Final AgenticState
            
        Returns:
            Response string
        """
        # If we have a direct answer (e.g., from early exit or execution)
        if state.answer:
            logger.info(f"Orchestrator: Using answer from state (length: {len(state.answer)} chars)")
            return state.answer
        
        # If we have tool results, try to extract answer
        if state.tool_results:
            # Look for RAG results
            for step_result in state.tool_results.values():
                if isinstance(step_result, dict):
                    result = step_result.get("result", "")
                    if result and isinstance(result, str):
                        return result
        
        # Fallback: generic response
        return "I've processed your request. If you need more specific information, please provide more details."
    
    def _handle_retry(self, state: AgenticState, max_retries: int = 3) -> AgenticState:
        """
        Handle retry logic based on evaluator feedback.
        
        Args:
            state: Current AgenticState
            max_retries: Maximum number of retries
            
        Returns:
            Updated AgenticState
        """
        retry_count = state.metadata.get("retry_count", 0)
        
        if retry_count >= max_retries:
            logger.warning(f"Orchestrator: Max retries ({max_retries}) reached")
            state.orchestration_status = OrchestrationStatus.FAILED
            return state
        
        evaluation = state.metadata.get("evaluation", {})
        if evaluation.get("should_retry", False):
            logger.info(f"Orchestrator: Retrying (attempt {retry_count + 1}/{max_retries})")
            state.metadata["retry_count"] = retry_count + 1
            # Reset execution plan to allow replanning
            state.execution_plan = None
            state.tool_results = {}
        
        return state
    
    def _handle_fallback(self, state: AgenticState) -> AgenticState:
        """
        Handle fallback mechanism.
        
        Args:
            state: Current AgenticState
            
        Returns:
            Updated AgenticState with fallback response
        """
        evaluation = state.metadata.get("evaluation", {})
        if evaluation.get("should_fallback", False):
            logger.info("Orchestrator: Using fallback mechanism")
            # Simple fallback: return a generic helpful message
            state.answer = (
                "I encountered some issues processing your request. "
                "Please try rephrasing your question or contact support for assistance."
            )
            state.orchestration_status = OrchestrationStatus.COMPLETED
        
        return state


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()

