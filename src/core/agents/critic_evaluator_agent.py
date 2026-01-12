"""
Critic / Evaluator Agent
Implements check & balance mechanism - evaluates tool correctness, parameter sanity, output consistency.
"""
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.orchestrator.state_schema import AgenticState, CriticFeedback
from settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_MODEL,
    CRITIC_EVALUATOR_SYSTEM_PROMPT,
    logger
)


class CriticEvaluatorAgent:
    """
    Critic / Evaluator Agent - Evaluates execution results.
    
    Purpose:
    - Evaluate tool correctness
    - Check parameter sanity
    - Verify output consistency
    - Detect business rule violations
    - Trigger retry or fallback if needed
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            model=AZURE_OPENAI_MODEL,
            temperature=0.3
        )
        
        self.system_prompt = CRITIC_EVALUATOR_SYSTEM_PROMPT
    
    def process(self, state: AgenticState) -> AgenticState:
        """
        Evaluate execution results and provide feedback.
        
        Args:
            state: AgenticState with execution results
            
        Returns:
            Updated AgenticState with critic_feedback
        """
        try:
            logger.info("Critic Evaluator: Evaluating execution results")
            
            # Gather context for evaluation
            user_query = state.user_query
            execution_plan = state.execution_plan or {}
            tool_results = state.tool_results or {}
            answer = state.answer or ""
            
            # Build evaluation context
            plan_summary = f"Goal: {execution_plan.get('goal', 'N/A')}\n"
            plan_summary += f"Steps: {len(execution_plan.get('steps', []))}"
            
            results_summary = "\n".join([
                f"Step {k}: {v.get('tool', 'N/A')} - Status: {v.get('status', 'unknown')}"
                for k, v in tool_results.items()
            ]) if tool_results else "No results"
            
            # Create prompt
            user_prompt = f"""Original User Query: {user_query}

Execution Plan:
{plan_summary}

Tool Execution Results:
{results_summary}

Generated Answer:
{answer}

Evaluate the quality and correctness of these results. Consider:
1. Were the right tools selected and executed?
2. Are the results accurate and complete?
3. Do the results logically address the user's query?
4. Are there any issues or inconsistencies?
5. Should we retry or use a fallback?

Provide your evaluation in the JSON format specified."""
            
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
                evaluation_data = json.loads(json_match.group())
            else:
                evaluation_data = json.loads(response_text)
            
            # Create CriticFeedback object
            critic_feedback = CriticFeedback(
                accuracy_score=evaluation_data.get("accuracy_score", 0.5),
                completeness_score=evaluation_data.get("completeness_score", 0.5),
                bias_detection_score=evaluation_data.get("bias_detection_score", 1.0),
                logical_consistency_score=evaluation_data.get("logical_consistency_score", 0.5),
                overall_quality_score=evaluation_data.get("overall_quality_score", 0.5),
                feedback_summary=evaluation_data.get("feedback_summary", ""),
                detailed_feedback=evaluation_data.get("detailed_feedback", [])
            )
            
            # Store in state
            state.critic_feedback = critic_feedback
            state.evaluation_scores = {
                "accuracy": critic_feedback.accuracy_score,
                "completeness": critic_feedback.completeness_score,
                "bias_detection": critic_feedback.bias_detection_score,
                "logical_consistency": critic_feedback.logical_consistency_score,
                "overall_quality": critic_feedback.overall_quality_score
            }
            
            # Check if retry or fallback is needed
            should_retry = evaluation_data.get("should_retry", False)
            should_fallback = evaluation_data.get("should_fallback", False)
            
            state.metadata["evaluation"] = {
                "should_retry": should_retry,
                "retry_reason": evaluation_data.get("retry_reason", ""),
                "should_fallback": should_fallback,
                "fallback_reason": evaluation_data.get("fallback_reason", ""),
                "overall_quality": critic_feedback.overall_quality_score
            }
            
            logger.info(f"Critic Evaluator: Overall quality score = {critic_feedback.overall_quality_score:.2f}")
            if should_retry:
                logger.warning(f"Critic Evaluator: Retry recommended - {evaluation_data.get('retry_reason', '')}")
            if should_fallback:
                logger.warning(f"Critic Evaluator: Fallback recommended - {evaluation_data.get('fallback_reason', '')}")
            
            return state
            
        except Exception as e:
            logger.error(f"Critic Evaluator Agent error: {e}", exc_info=True)
            # Fallback: create default feedback
            state.critic_feedback = CriticFeedback(
                accuracy_score=0.5,
                completeness_score=0.5,
                bias_detection_score=1.0,
                logical_consistency_score=0.5,
                overall_quality_score=0.5,
                feedback_summary=f"Evaluation error: {str(e)}",
                detailed_feedback=[]
            )
            state.metadata["evaluation"] = {"error": str(e)}
            return state


def critic_evaluator_node(state: AgenticState) -> AgenticState:
    """
    LangGraph node function for Critic Evaluator Agent.
    
    Args:
        state: AgenticState
        
    Returns:
        Updated AgenticState
    """
    agent = CriticEvaluatorAgent()
    return agent.process(state)

