"""
Execution Agent - ReAct Agent powered by LLM
LLM-powered agent that executes tools from execution plan with reasoning.
Agent has access to execution steps and limited tools, reasons about execution sequence.
"""
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
    EXECUTION_AGENT_SYSTEM_PROMPT,
    logger
)


class ExecutionAgent:
    """
    Execution Agent - ReAct Agent that executes tools with LLM reasoning.
    
    The agent:
    1. Gets execution plan from state
    2. Gets available tools (limited to execution plan)
    3. Uses LLM to reason about execution sequence
    4. Calls tools based on reasoning
    5. Observes results and adapts
    6. Returns desired output
    
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
                    cls._instance = super(ExecutionAgent, cls).__new__(cls)
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
            temperature=0.3
        )
        ExecutionAgent._initialized = True
    
    def _get_available_tools(
        self,
        execution_plan: Dict[str, Any],
        tool_registry: Dict[str, List[BaseTool]]
    ) -> tuple[List[BaseTool], Dict[str, BaseTool]]:
        """
        Get available tools for execution.
        Limits tools to those mentioned in execution plan + domain tools.
        """
        steps = execution_plan.get("steps", [])
        required_tool_names = {step.get("tool") for step in steps if step.get("tool")}
        
        # Get domains from tool names and intent classification
        required_domains = {tool_name.split(".")[0] if "." in tool_name else tool_name.split("_")[0] 
                           for tool_name in required_tool_names}
        
        # Collect tools from required domains
        available_tools = []
        tool_map = {}
        
        for domain in required_domains:
            if domain in tool_registry:
                for tool in tool_registry[domain]:
                    tool_map[tool.name] = tool
                    available_tools.append(tool)
        
        # Always include RAG tools
        if "rag" in tool_registry:
            for tool in tool_registry["rag"]:
                if tool.name not in tool_map:
                    tool_map[tool.name] = tool
                    available_tools.append(tool)
        
        logger.info(f"Execution Agent: Providing {len(available_tools)} tools to LLM agent")
        return available_tools, tool_map
    
    def _build_execution_plan_context(self, execution_plan: Dict[str, Any]) -> str:
        """Build execution plan context for LLM."""
        goal = execution_plan.get("goal", "Execute the planned steps")
        steps = execution_plan.get("steps", [])
        
        context = f"""Execution Plan:

Goal: {goal}

Steps to Execute:
"""
        for step in steps:
            step_id = step.get("step_id")
            tool_name = step.get("tool")
            description = step.get("description", "")
            inputs = step.get("inputs", {})
            depends_on = step.get("depends_on", [])
            
            context += f"""
Step {step_id}:
  - Tool: {tool_name}
  - Description: {description}
  - Inputs: {inputs}
  - Depends on: {depends_on if depends_on else 'None'}
"""
        
        return context
    
    def _build_tools_context(self, tools: List[BaseTool]) -> str:
        """Build tools description for LLM."""
        context = "Available Tools:\n"
        for tool in tools:
            # Get tool schema if available
            try:
                args_schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
                params = args_schema.get("properties", {})
                param_desc = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in params.items()])
            except:
                param_desc = "see tool description"
            
            context += f"\n- {tool.name}\n"
            context += f"  Description: {tool.description}\n"
            if param_desc:
                context += f"  Parameters: {param_desc}\n"
        
        return context
    
    def _execute_with_react(
        self,
        execution_plan: Dict[str, Any],
        tools: List[BaseTool],
        tool_map: Dict[str, BaseTool],
        user_query: str,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Execute using ReAct pattern with LLM.
        LLM reasons about execution and calls tools to achieve the goal.
        """
        goal = execution_plan.get("goal", "Execute the planned steps")
        steps = execution_plan.get("steps", [])
        
        # Build context
        plan_context = self._build_execution_plan_context(execution_plan)
        tools_context = self._build_tools_context(tools)
        
        # System prompt with execution plan and tools
        system_prompt = f"""{EXECUTION_AGENT_SYSTEM_PROMPT}

{plan_context}

{tools_context}

Your Task:
Execute the execution plan to achieve the goal: "{goal}"

Instructions:
1. Review the execution plan steps above
2. Use the available tools to execute each step
3. Follow the step order, but you can reason about each step before executing
4. If a step depends on previous steps, use the results from those steps
5. After each tool call, observe the result and reason about next steps
6. Continue until all steps are executed or goal is achieved
7. Provide a final summary of execution results

ReAct Format:
Thought: [Your reasoning about what to do]
Action: [Tool name to call]
Action Input: [Tool parameters]
Observation: [Tool result]
Thought: [Reason about result and next step]
... (repeat until complete)

Begin execution now."""
        
        # Conversation history for ReAct loop
        messages = [SystemMessage(content=system_prompt)]
        
        # Initial user query
        user_prompt = f"""User Query: {user_query}

Execute the execution plan step by step using the available tools. 
Reason about each step, call the appropriate tool, observe results, and continue."""
        
        messages.append(HumanMessage(content=user_prompt))
        
        execution_results = {}
        step_results = {}
        executed_steps = []
        iteration = 0
        
        # ReAct loop
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Get LLM response (with tool calling if available)
                try:
                    # Try structured tool calling
                    llm_with_tools = self.llm.bind_tools(tools) if hasattr(self.llm, 'bind_tools') else self.llm
                    response = llm_with_tools.invoke(messages)
                except:
                    response = self.llm.invoke(messages)
                
                # Add LLM response to conversation
                messages.append(response)
                
                # Check if LLM wants to call a tool
                tool_calls_made = False
                
                # Handle structured tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                        
                        tool_to_execute = tool_map.get(tool_name)
                        if tool_to_execute:
                            logger.info(f"Execution Agent: LLM calling tool {tool_name} with args {tool_args}")
                            
                            try:
                                # Execute tool
                                tool_result = tool_to_execute.invoke(tool_args)
                                
                                # Store result
                                # Find which step this tool belongs to
                                step_id = None
                                for step in steps:
                                    if step.get("tool") == tool_name or tool_name in step.get("tool", ""):
                                        step_id = step.get("step_id")
                                        break
                                
                                if step_id:
                                    execution_results[f"step_{step_id}"] = {
                                        "tool": tool_name,
                                        "result": tool_result,
                                        "status": "success"
                                    }
                                    step_results[f"step_{step_id}"] = tool_result
                                    executed_steps.append(step_id)
                                
                                # Add tool result as observation
                                observation = f"Tool {tool_name} executed successfully. Result: {str(tool_result)[:1000]}"
                                # Use HumanMessage for tool results (ToolMessage may not be available in all versions)
                                try:
                                    from langchain_core.messages import ToolMessage
                                    messages.append(ToolMessage(content=observation, tool_call_id=tool_call.get('id', '')))
                                except:
                                    messages.append(HumanMessage(content=f"Observation: {observation}"))
                                
                                tool_calls_made = True
                                logger.info(f"Execution Agent: Tool {tool_name} executed successfully")
                                
                            except Exception as e:
                                error_msg = f"Error executing {tool_name}: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                try:
                                    from langchain_core.messages import ToolMessage
                                    messages.append(ToolMessage(
                                        content=f"Error: {error_msg}",
                                        tool_call_id=tool_call.get('id', '')
                                    ))
                                except:
                                    messages.append(HumanMessage(content=f"Observation: Error: {error_msg}"))
                                
                                if step_id:
                                    execution_results[f"step_{step_id}"] = {
                                        "tool": tool_name,
                                        "result": None,
                                        "status": "error",
                                        "error": str(e)
                                    }
                
                # Handle text-based tool calls (fallback)
                if not tool_calls_made:
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Check if LLM mentioned a tool in response
                    for tool in tools:
                        if tool.name.lower() in response_text.lower():
                            # Try to extract parameters from response
                            tool_inputs = {}
                            
                            # Look for JSON in response
                            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    tool_inputs = json.loads(json_match.group())
                                except:
                                    pass
                            
                            # If no JSON, try to get from step inputs
                            if not tool_inputs:
                                for step in steps:
                                    if step.get("tool") == tool.name or tool.name in step.get("tool", ""):
                                        tool_inputs = step.get("inputs", {}).copy()
                                        # Resolve dependencies
                                        for dep_step_id in step.get("depends_on", []):
                                            dep_key = f"step_{dep_step_id}"
                                            if dep_key in step_results:
                                                dep_result = step_results[dep_key]
                                                if isinstance(dep_result, dict):
                                                    tool_inputs.update(dep_result)
                                        break
                            
                            logger.info(f"Execution Agent: Executing {tool.name} (text-based call)")
                            
                            try:
                                tool_result = tool.invoke(tool_inputs)
                                
                                # Store result
                                for step in steps:
                                    if step.get("tool") == tool.name or tool.name in step.get("tool", ""):
                                        step_id = step.get("step_id")
                                        execution_results[f"step_{step_id}"] = {
                                            "tool": tool.name,
                                            "result": tool_result,
                                            "status": "success"
                                        }
                                        step_results[f"step_{step_id}"] = tool_result
                                        executed_steps.append(step_id)
                                        break
                                
                                # Add observation
                                observation = f"Tool {tool.name} executed. Result: {str(tool_result)[:1000]}"
                                messages.append(HumanMessage(content=f"Observation: {observation}"))
                                
                                tool_calls_made = True
                                break
                                
                            except Exception as e:
                                logger.error(f"Error executing {tool.name}: {e}", exc_info=True)
                                messages.append(HumanMessage(content=f"Observation: Error executing {tool.name}: {str(e)}"))
                
                # Check if all steps are complete
                if len(executed_steps) >= len(steps):
                    logger.info("Execution Agent: All steps completed")
                    break
                
                # Check if LLM says it's done
                if hasattr(response, 'content'):
                    response_text = response.content.lower()
                    if any(phrase in response_text for phrase in ["final answer", "all steps complete", "execution complete", "done"]):
                        logger.info("Execution Agent: LLM indicated completion")
                        break
                
            except Exception as e:
                logger.error(f"Execution Agent: Error in ReAct iteration {iteration}: {e}", exc_info=True)
                break
        
        # Get final answer from LLM if available
        final_answer = None
        if messages and hasattr(messages[-1], 'content'):
            final_answer = messages[-1].content
        
        return {
            "execution_results": execution_results,
            "executed_steps": executed_steps,
            "total_steps": len(steps),
            "final_answer": final_answer,
            "iterations": iteration
        }
    
    @trace_agent("execution_agent")
    def process(self, state: AgenticState, tool_registry: Dict[str, List[BaseTool]], node_registry: Any) -> AgenticState:
        """
        Execute tools from execution plan using ReAct agent.
        Gets execution plan and tools from state, lets LLM reason and execute.
        """
        try:
            logger.info("Execution Agent (ReAct): Starting LLM-powered execution")
            
            if not state.execution_plan:
                logger.warning("Execution Agent: No execution plan found")
                return state
            
            execution_plan = state.execution_plan
            
            # Get available tools for this execution plan
            available_tools, tool_map = self._get_available_tools(execution_plan, tool_registry)
            
            if not available_tools:
                logger.warning("Execution Agent: No tools available")
                state.metadata["execution"] = {"error": "No tools available"}
                return state
            
            # Execute using ReAct pattern
            react_result = self._execute_with_react(
                execution_plan=execution_plan,
                tools=available_tools,
                tool_map=tool_map,
                user_query=state.normalized_query or state.user_query
            )
            
            # Store results in state
            state.tool_results = react_result["execution_results"]
            
            # Store final answer if LLM provided one
            if react_result.get("final_answer"):
                # Extract structured results for response generator
                state.metadata["execution"] = {
                    "executed_steps": react_result["executed_steps"],
                    "total_steps": react_result["total_steps"],
                    "success_rate": len(react_result["executed_steps"]) / react_result["total_steps"] if react_result["total_steps"] > 0 else 0,
                    "execution_mode": "react",
                    "iterations": react_result["iterations"],
                    "final_answer": react_result["final_answer"]
                }
            else:
                state.metadata["execution"] = {
                    "executed_steps": react_result["executed_steps"],
                    "total_steps": react_result["total_steps"],
                    "success_rate": len(react_result["executed_steps"]) / react_result["total_steps"] if react_result["total_steps"] > 0 else 0,
                    "execution_mode": "react",
                    "iterations": react_result["iterations"]
                }
            
            logger.info(f"Execution Agent (ReAct): Completed {len(react_result['executed_steps'])}/{react_result['total_steps']} steps in {react_result['iterations']} iterations")
            return state
            
        except Exception as e:
            logger.error(f"Execution Agent (ReAct) error: {e}", exc_info=True)
            state.metadata["execution"] = {"error": str(e), "execution_mode": "react"}
            return state


def execution_node(state: AgenticState, tool_registry: Dict[str, List[BaseTool]], node_registry: Any) -> AgenticState:
    """LangGraph node function for Execution Agent."""
    agent = ExecutionAgent()
    return agent.process(state, tool_registry, node_registry)
