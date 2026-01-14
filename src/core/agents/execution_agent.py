"""
Execution Agent - ReAct Agent powered by LLM
LLM-powered agent that executes tools from execution plan with reasoning.
Agent has access to execution steps and limited tools, reasons about execution sequence.
Uses LangChain's create_react_agent for proper ReAct execution.
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

# Try LangGraph's create_react_agent first (better for chat models with tool calling)
try:
    from langgraph.prebuilt import create_react_agent as langgraph_create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    langgraph_create_react_agent = None
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph create_react_agent not available. Trying LangChain version.")

# Fallback to LangChain's older create_react_agent (for text completion models)
try:
    from langchain.agents import create_react_agent, AgentExecutor
    LANGCHAIN_AGENTS_AVAILABLE = True
except ImportError:
    try:
        from langchain.agents.react.agent import create_react_agent
        from langchain.agents import AgentExecutor
        LANGCHAIN_AGENTS_AVAILABLE = True
    except ImportError:
        create_react_agent = None
        AgentExecutor = None
        LANGCHAIN_AGENTS_AVAILABLE = False
        logger.warning("LangChain create_react_agent not available. Will use manual ReAct execution.")

try:
    from langchain import hub
except ImportError:
    hub = None
    logger.warning("LangChain hub not available. Will use custom prompts.")


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
    
    def _truncate_tool_names(self, tools: List[BaseTool]) -> tuple[List[BaseTool], Dict[str, BaseTool]]:
        """
        Truncate tool names to 64 characters (Azure OpenAI limit).
        
        Args:
            tools: List of tools with potentially long names
            
        Returns:
            Tuple of (truncated_tools, name_mapping) where:
            - truncated_tools: List of tools with truncated names
            - name_mapping: Dict mapping truncated names to original tool objects
        """
        max_name_length = 64
        truncated_tools = []
        name_mapping = {}  # Map truncated names to original tool objects
        
        for tool in tools:
            original_name = tool.name
            if len(original_name) > max_name_length:
                # Truncate name
                if '.' in original_name:
                    domain_part, api_part = original_name.split('.', 1)
                    available_for_api = max_name_length - len(domain_part) - 1
                    if available_for_api > 0:
                        truncated_name = f"{domain_part}.{api_part[:available_for_api]}"
                    else:
                        truncated_name = original_name[:max_name_length]
                else:
                    truncated_name = original_name[:max_name_length]
                
                # Create a copy of the tool with truncated name
                from langchain_core.tools import StructuredTool
                truncated_tool = StructuredTool(
                    name=truncated_name,
                    description=tool.description,
                    func=tool.func,
                    args_schema=tool.args_schema,
                    coroutine=tool.coroutine if hasattr(tool, 'coroutine') else None
                )
                truncated_tool._original_name = original_name
                truncated_tools.append(truncated_tool)
                name_mapping[truncated_name] = tool
                logger.debug(f"Truncated tool name '{original_name}' ({len(original_name)} chars) to '{truncated_name}' ({len(truncated_name)} chars)")
            else:
                truncated_tools.append(tool)
                name_mapping[original_name] = tool
        
        return truncated_tools, name_mapping
    
    def _execute_with_react(
        self,
        execution_plan: Dict[str, Any],
        tools: List[BaseTool],
        tool_map: Dict[str, BaseTool],
        user_query: str,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Execute using LangChain's create_react_agent for proper ReAct execution.
        Uses LangChain's built-in ReAct agent executor instead of manual loop.
        """
        goal = execution_plan.get("goal", "Execute the planned steps")
        steps = execution_plan.get("steps", [])
        
        # Build context
        plan_context = self._build_execution_plan_context(execution_plan)
        tools_context = self._build_tools_context(tools)
        
        # Enhanced system prompt with execution plan and tools
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

Remember: You are executing banking operations - be precise, validate inputs, and handle errors gracefully."""
        
        # Create prompt template for ReAct agent
        # Try to pull from hub, fallback to custom prompt
        prompt = None
        if hub is not None:
            try:
                prompt = hub.pull("hwchase17/react")
                logger.info("Successfully pulled react prompt from LangChain hub")
            except Exception as e:
                logger.warning(f"Could not pull react prompt from hub: {e}. Using custom prompt.")
        
        if prompt is None:
            # Create custom prompt using PromptTemplate
            # create_react_agent expects prompts with 'tools', 'tool_names', 'input', and 'agent_scratchpad' variables
            from langchain_core.prompts import PromptTemplate
            
            # Escape all curly braces in system_prompt to prevent them from being interpreted as template variables
            # Replace { with {{ and } with }} so they become literal braces in the template
            escaped_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
            
            # Build the template string without using f-string for template variables
            # This ensures {tools}, {tool_names}, {input}, {agent_scratchpad} are treated as template variables
            # while escaped_system_prompt content is treated as literal text
            prompt_template_str = escaped_system_prompt + """

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
            
            prompt = PromptTemplate.from_template(prompt_template_str)
        
        # Try LangGraph's create_react_agent first (better for chat models with tool calling)
        if LANGGRAPH_AVAILABLE:
            try:
                logger.info("Using LangGraph create_react_agent (chat model with tool calling)")
                return self._execute_with_langgraph_react(
                    execution_plan, tools, tool_map, user_query, system_prompt, max_iterations
                )
            except Exception as e:
                logger.warning(f"LangGraph create_react_agent failed: {e}. Trying LangChain version.", exc_info=True)
        
        # Fallback to LangChain's create_react_agent (for text completion models)
        if LANGCHAIN_AGENTS_AVAILABLE and create_react_agent is not None:
            try:
                logger.info("Using LangChain create_react_agent (text completion model)")
                return self._execute_with_langchain_react(
                    execution_plan, tools, tool_map, user_query, prompt, max_iterations
                )
            except Exception as e:
                logger.error(f"LangChain create_react_agent failed: {e}. Falling back to manual execution.", exc_info=True)
        
        # Final fallback to manual execution
        logger.warning("No create_react_agent available. Using manual ReAct execution.")
        return self._execute_with_manual_react(execution_plan, tools, tool_map, user_query, max_iterations)
    
    def _execute_with_langgraph_react(
        self,
        execution_plan: Dict[str, Any],
        tools: List[BaseTool],
        tool_map: Dict[str, BaseTool],
        user_query: str,
        system_prompt: str,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Execute using LangGraph's create_react_agent (for chat models with tool calling).
        This is the recommended approach for chat models like AzureChatOpenAI.
        """
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        
        goal = execution_plan.get("goal", "Execute the planned steps")
        steps = execution_plan.get("steps", [])
        
        try:
            # Truncate tool names to 64 characters (Azure OpenAI limit)
            truncated_tools, tool_name_mapping = self._truncate_tool_names(tools)
            
            # Create LangGraph agent with system prompt (using truncated tools)
            graph = langgraph_create_react_agent(
                self.llm,
                truncated_tools,
                prompt=system_prompt
            )
            
            # Prepare input messages
            input_messages = [HumanMessage(content=user_query)]
            
            # Invoke the graph
            logger.info(f"Invoking LangGraph agent with {len(tools)} tools")
            result = graph.invoke({
                "messages": input_messages
            })
            
            # Extract messages from result
            messages = result.get("messages", [])
            
            # Process messages to extract tool execution results
            execution_results = {}
            step_results = {}
            executed_steps = []
            tool_call_count = 0
            
            # Track tool calls and their results
            tool_call_map = {}  # Maps tool_call_id to tool_name
            
            for message in messages:
                # Process AIMessage with tool_calls
                if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_call_id = tool_call.get('id', '')
                        tool_name = tool_call.get('name', '')
                        tool_call_map[tool_call_id] = tool_name
                        tool_call_count += 1
                        logger.debug(f"Tool call: {tool_name} (id: {tool_call_id})")
                
                # Process ToolMessage (tool results)
                elif isinstance(message, ToolMessage):
                    tool_call_id = message.tool_call_id if hasattr(message, 'tool_call_id') else ''
                    tool_name = tool_call_map.get(tool_call_id, message.name if hasattr(message, 'name') else '')
                    # Map truncated tool name back to original if needed
                    original_tool = tool_name_mapping.get(tool_name)
                    if original_tool:
                        # Check if the tool object has _original_name attribute (from truncated tool)
                        # Otherwise, the tool_name is already the original
                        if hasattr(original_tool, '_original_name'):
                            original_tool_name = original_tool._original_name
                        else:
                            # Check if any truncated tool has this as its truncated name
                            for truncated_tool in truncated_tools:
                                if truncated_tool.name == tool_name and hasattr(truncated_tool, '_original_name'):
                                    original_tool_name = truncated_tool._original_name
                                    break
                            else:
                                original_tool_name = tool_name
                    else:
                        original_tool_name = tool_name
                    tool_result = message.content if hasattr(message, 'content') else str(message)
                    
                    # Try to parse JSON result
                    if isinstance(tool_result, str):
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, dict):
                                tool_result = parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                    
                    # Log tool result with detailed information
                    status = "unknown"
                    status_code = None
                    if isinstance(tool_result, dict):
                        status = tool_result.get("status", "unknown")
                        status_code = tool_result.get("status_code")
                        if status_code and 200 <= status_code < 300:
                            logger.info(f"✅ Tool '{tool_name}' executed successfully (HTTP {status_code})")
                            # Log the actual data returned
                            data = tool_result.get("data", {})
                            if data:
                                logger.info(f"📊 Tool '{tool_name}' Response Data: {json.dumps(data, indent=2, default=str)[:1000]}...")
                            else:
                                logger.info(f"📊 Tool '{tool_name}' Response: {json.dumps(tool_result, indent=2, default=str)[:1000]}...")
                        elif status == "error":
                            logger.error(f"❌ Tool '{tool_name}' failed: {tool_result.get('error', 'Unknown error')}")
                        else:
                            logger.warning(f"⚠️ Tool '{tool_name}' returned status: {status}")
                    else:
                        logger.info(f"📊 Tool '{tool_name}' Result: {str(tool_result)[:500]}...")
                    
                    # Find which step this tool belongs to
                    step_id = None
                    for plan_step in steps:
                        if plan_step.get("tool") == tool_name or tool_name in plan_step.get("tool", ""):
                            step_id = plan_step.get("step_id")
                            break
                    
                    if step_id:
                        execution_results[f"step_{step_id}"] = {
                            "tool": original_tool_name,  # Use original name in results
                            "result": tool_result,
                            "status": "success" if not isinstance(tool_result, dict) or tool_result.get("status") != "error" else "error"
                        }
                        step_results[f"step_{step_id}"] = tool_result
                        if step_id not in executed_steps:
                            executed_steps.append(step_id)
            
            # Get final answer from last AI message without tool calls
            final_answer = ""
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    if hasattr(message, 'content') and message.content:
                        # Check if this message has tool calls
                        has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                        if not has_tool_calls:
                            final_answer = message.content
                            break
            
            # Sanitize final answer: remove API endpoints and technical details
            if final_answer:
                final_answer = self._sanitize_response(final_answer)
            
            # Log final answer from LLM
            if final_answer:
                logger.info("=" * 80)
                logger.info("🤖 LLM FINAL ANSWER:")
                logger.info("=" * 80)
                logger.info(final_answer)
                logger.info("=" * 80)
            else:
                logger.warning("⚠️ No final answer generated by LLM")
            
            logger.info(f"LangGraph execution completed: {len(executed_steps)}/{len(steps)} steps, {tool_call_count} tool calls")
            
            return {
                "execution_results": execution_results,
                "executed_steps": executed_steps,
                "total_steps": len(steps),
                "final_answer": final_answer,
                "iterations": tool_call_count
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph ReAct execution: {e}", exc_info=True)
            raise
    
    def _sanitize_response(self, response: str) -> str:
        """
        Sanitize response to remove API endpoints and technical details.
        
        Args:
            response: Raw response string
            
        Returns:
            Sanitized response string
        """
        import re
        
        # Remove API endpoint URLs (but keep user-facing URLs like approval links)
        # Pattern: https://api.*.paypal.com/... or similar API endpoints
        api_endpoint_pattern = r'https://api[^\s\)]+(?:/v\d+[^\s\)]*)?'
        
        # Check if URL is an API endpoint (contains /v1/, /v2/, etc. after domain)
        def is_api_endpoint(url: str) -> bool:
            # User-facing URLs typically don't have /v1/ or /v2/ in path
            # or are www.paypal.com (approval links)
            if 'www.paypal.com' in url or 'sandbox.paypal.com/checkoutnow' in url:
                return False
            if '/v1/' in url or '/v2/' in url or '/api/' in url:
                return True
            return False
        
        # Find and remove API endpoints
        def replace_api_endpoint(match):
            url = match.group(0)
            if is_api_endpoint(url):
                return ""  # Remove API endpoints
            return url  # Keep user-facing URLs
        
        sanitized = re.sub(api_endpoint_pattern, replace_api_endpoint, response)
        
        # Remove lines that are just API endpoint references
        lines = sanitized.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just API endpoints
            if stripped and not re.match(r'^https://api[^\s\)]+(?:/v\d+[^\s\)]*)?$', stripped):
                filtered_lines.append(line)
            elif not stripped:  # Keep empty lines for formatting
                filtered_lines.append(line)
        
        sanitized = '\n'.join(filtered_lines)
        
        # Clean up multiple blank lines
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        return sanitized.strip()
    
    def _execute_with_langchain_react(
        self,
        execution_plan: Dict[str, Any],
        tools: List[BaseTool],
        tool_map: Dict[str, BaseTool],
        user_query: str,
        prompt: Any,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Execute using LangChain's create_react_agent (for text completion models).
        This is the older approach, kept as fallback.
        """
        goal = execution_plan.get("goal", "Execute the planned steps")
        steps = execution_plan.get("steps", [])
        
        try:
            # Truncate tool names to 64 characters (Azure OpenAI limit)
            truncated_tools, tool_name_mapping = self._truncate_tool_names(tools)
            
            # Create ReAct agent (using truncated tools)
            agent = create_react_agent(self.llm, truncated_tools, prompt)
            
            # Create agent executor (using truncated tools)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=truncated_tools,
                verbose=True,
                max_iterations=max_iterations,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            # Build input prompt
            input_prompt = f"""User Query: {user_query}

Execute the execution plan step by step using the available tools. 
Reason about each step, call the appropriate tool, observe results, and continue."""
            
            execution_results = {}
            step_results = {}
            executed_steps = []
            
            # Invoke agent executor
            result = agent_executor.invoke({
                "input": input_prompt
            })
            
            # Extract execution results from agent execution
            final_answer = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Process intermediate steps to extract tool execution results
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    
                    tool_name = action.tool if hasattr(action, 'tool') else str(action)
                    tool_input = action.tool_input if hasattr(action, 'tool_input') else {}
                    
                    # Map truncated tool name back to original if needed
                    original_tool = tool_name_mapping.get(tool_name)
                    if original_tool:
                        # Check if any truncated tool has this as its truncated name
                        for truncated_tool in truncated_tools:
                            if truncated_tool.name == tool_name and hasattr(truncated_tool, '_original_name'):
                                original_tool_name = truncated_tool._original_name
                                break
                        else:
                            original_tool_name = tool_name
                    else:
                        original_tool_name = tool_name
                    
                    # Find which step this tool belongs to (check both truncated and original names)
                    step_id = None
                    for plan_step in steps:
                        plan_tool_name = plan_step.get("tool", "")
                        if plan_tool_name == original_tool_name or original_tool_name in plan_tool_name:
                            step_id = plan_step.get("step_id")
                            break
                        elif plan_tool_name == tool_name or tool_name in plan_tool_name:
                            step_id = plan_step.get("step_id")
                            break
                    
                    if step_id:
                        execution_results[f"step_{step_id}"] = {
                            "tool": original_tool_name,  # Use original name in results
                            "result": observation,
                            "status": "success" if not isinstance(observation, dict) or observation.get("status") != "error" else "error"
                        }
                        step_results[f"step_{step_id}"] = observation
                        if step_id not in executed_steps:
                            executed_steps.append(step_id)
            
            return {
                "execution_results": execution_results,
                "executed_steps": executed_steps,
                "total_steps": len(steps),
                "final_answer": final_answer,
                "iterations": len(intermediate_steps)
            }
            
        except Exception as e:
            logger.error(f"Error in LangChain ReAct execution: {e}", exc_info=True)
            raise
    
    def _execute_with_manual_react(
        self,
        execution_plan: Dict[str, Any],
        tools: List[BaseTool],
        tool_map: Dict[str, BaseTool],
        user_query: str,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Fallback manual ReAct execution if create_react_agent fails.
        This is a simplified version of the original manual loop.
        """
        goal = execution_plan.get("goal", "Execute the planned steps")
        steps = execution_plan.get("steps", [])
        
        # Build context
        plan_context = self._build_execution_plan_context(execution_plan)
        tools_context = self._build_tools_context(tools)
        
        # System prompt
        system_prompt = f"""{EXECUTION_AGENT_SYSTEM_PROMPT}

{plan_context}

{tools_context}

Your Task: Execute the execution plan to achieve the goal: "{goal}"
"""
        
        messages = [SystemMessage(content=system_prompt)]
        user_prompt = f"""User Query: {user_query}

Execute the execution plan step by step using the available tools."""
        messages.append(HumanMessage(content=user_prompt))
        
        execution_results = {}
        step_results = {}
        executed_steps = []
        iteration = 0
        
        # Truncate tool names to 64 characters (Azure OpenAI limit) before binding
        truncated_tools, tool_name_mapping = self._truncate_tool_names(tools)
        
        # Update tool_map to use truncated names for lookup
        truncated_tool_map = {}
        for truncated_name, original_tool in tool_name_mapping.items():
            truncated_tool_map[truncated_name] = original_tool
        
        # Simplified manual loop as fallback
        while iteration < max_iterations:
            iteration += 1
            try:
                llm_with_tools = self.llm.bind_tools(truncated_tools) if hasattr(self.llm, 'bind_tools') else self.llm
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', '')  # This will be truncated name
                        tool_args = tool_call.get('args', {})
                        # Use truncated_tool_map to get the original tool
                        # Also check tool_map in case it uses original names
                        tool_to_execute = truncated_tool_map.get(tool_name) or tool_map.get(tool_name)
                        
                        # If still not found, try to find by original name from mapping
                        if not tool_to_execute:
                            # Check if any tool has this as truncated name
                            for orig_tool in tools:
                                if hasattr(orig_tool, '_original_name') and orig_tool._original_name == tool_name:
                                    tool_to_execute = orig_tool
                                    break
                                elif orig_tool.name == tool_name:
                                    tool_to_execute = orig_tool
                                    break
                        
                        if tool_to_execute:
                            try:
                                tool_result = tool_to_execute.invoke(tool_args)
                                
                                # Find step
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
                                
                                from langchain_core.messages import ToolMessage
                                messages.append(ToolMessage(
                                    content=str(tool_result)[:1000],
                                    tool_call_id=tool_call.get('id', '')
                                ))
                            except Exception as e:
                                logger.error(f"Error executing {tool_name}: {e}", exc_info=True)
                                from langchain_core.messages import ToolMessage
                                messages.append(ToolMessage(
                                    content=f"Error: {str(e)}",
                                    tool_call_id=tool_call.get('id', '')
                                ))
                
                if len(executed_steps) >= len(steps):
                    break
                    
            except Exception as e:
                logger.error(f"Error in manual ReAct iteration: {e}", exc_info=True)
                break
        
        final_answer = messages[-1].content if messages and hasattr(messages[-1], 'content') else None
        
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
            
            # Log available tools for debugging
            tool_names = [tool.name for tool in available_tools]
            logger.info(f"Execution Agent: Available tools ({len(available_tools)}): {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}")
            
            # Log which agent implementation will be used
            if LANGGRAPH_AVAILABLE:
                logger.info("Execution Agent: Will use LangGraph create_react_agent (chat model with tool calling)")
            elif LANGCHAIN_AGENTS_AVAILABLE:
                logger.info("Execution Agent: Will use LangChain create_react_agent (text completion model)")
            else:
                logger.warning("Execution Agent: Will use manual ReAct execution (no create_react_agent available)")
            
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
                final_answer_text = react_result["final_answer"]
                # Store in state answer field
                state.answer = final_answer_text
                
                # Log the final answer being stored
                logger.info("=" * 80)
                logger.info("💾 STORING FINAL ANSWER IN STATE:")
                logger.info("=" * 80)
                logger.info(final_answer_text)
                logger.info("=" * 80)
                
                # Extract structured results for response generator
                state.metadata["execution"] = {
                    "executed_steps": react_result["executed_steps"],
                    "total_steps": react_result["total_steps"],
                    "success_rate": len(react_result["executed_steps"]) / react_result["total_steps"] if react_result["total_steps"] > 0 else 0,
                    "execution_mode": "react",
                    "iterations": react_result["iterations"],
                    "final_answer": final_answer_text
                }
            else:
                logger.warning("⚠️ No final answer from ReAct execution to store in state")
                state.metadata["execution"] = {
                    "executed_steps": react_result["executed_steps"],
                    "total_steps": react_result["total_steps"],
                    "success_rate": len(react_result["executed_steps"]) / react_result["total_steps"] if react_result["total_steps"] > 0 else 0,
                    "execution_mode": "react",
                    "iterations": react_result["iterations"]
                }
            
            # Log summary of tool execution results
            if state.tool_results:
                logger.info("=" * 80)
                logger.info("📋 TOOL EXECUTION SUMMARY:")
                logger.info("=" * 80)
                for step_key, step_result in state.tool_results.items():
                    tool_name = step_result.get("tool", "unknown")
                    status = step_result.get("status", "unknown")
                    result = step_result.get("result", {})
                    if isinstance(result, dict):
                        status_code = result.get("status_code")
                        if status_code:
                            logger.info(f"  {step_key}: {tool_name} - Status: {status} (HTTP {status_code})")
                        else:
                            logger.info(f"  {step_key}: {tool_name} - Status: {status}")
                    else:
                        logger.info(f"  {step_key}: {tool_name} - Status: {status}")
                logger.info("=" * 80)
            
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
