"""
Simplified Tool Registry
Registers API endpoints as LangChain tools and stores them in a file.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model
from typing import Union, Optional
import httpx
from settings import (
    logger,
    RATE_LIMIT_TOOL_REGISTRY_CAPACITY,
    RATE_LIMIT_TOOL_REGISTRY_REFILL,
    BANKING_API_BASE_URL
)
from src.core.utils.rate_limiter import check_rate_limit


class APISpec(BaseModel):
    """Simple API specification"""
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE, PATCH")
    url: str = Field(description="API endpoint URL with optional path parameters like {account_id}")
    headers: Dict[str, str] = Field(default_factory=dict)
    request_schema: Dict[str, Any] = Field(default_factory=dict, description="Parameter schema: {param: {type, required, default, description}}")
    query_params: List[str] = Field(default_factory=list)


class ToolRegistry:
    """Simple tool registry - register API endpoints as LangChain tools and save to file."""
    
    def __init__(self, persistence_file: Optional[str] = None):
        self._domain_tools: Dict[str, List[BaseTool]] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Set persistence file path
        if persistence_file is None:
            tools_dir = Path(__file__).parent.parent / "tools"
            tools_dir.mkdir(exist_ok=True)
            persistence_file = str(tools_dir / "registered_banking_tools.json")
        
        self._persistence_file = persistence_file
        self._register_default_tools()
        self._load_from_file()
    
    def _register_default_tools(self):
        """Register default System tools (only system search, RAG removed per requirements)"""
        # Register System Search Tool
        try:
            from src.core.tools.system_search import system_search_tool
            self.register_tool("system", system_search_tool, {
                "name": "system.search",
                "description": "Search system capabilities, tools, logs, or status",
                "category": "system"
            }, save_to_file=False)  # Don't save system tools to file
        except Exception as e:
            logger.warning(f"Could not register System Search tool: {e}")
    
    def _create_api_function(
        self, tool_name: str, description: str, api_spec: APISpec,
        auth_token: Optional[str] = None, timeout: int = 30,
        use_oauth2_auth: bool = False
    ) -> Callable:
        """Create a function that calls an HTTP API endpoint."""
        # Extract path parameters, but exclude {{base_url}} placeholder
        # Replace {{base_url}} temporarily to avoid matching {base_url} inside it
        temp_url = api_spec.url.replace("{{base_url}}", "__BASE_URL_PLACEHOLDER__")
        path_params = re.findall(r'\{(\w+)\}', temp_url)
        method_map = {
            "GET": lambda c, u, h, q, b: c.get(u, headers=h, params=q),
            "POST": lambda c, u, h, q, b: c.post(u, headers=h, json=b, params=q),
            "PUT": lambda c, u, h, q, b: c.put(u, headers=h, json=b, params=q),
            "DELETE": lambda c, u, h, q, b: c.delete(u, headers=h, params=q),
            "PATCH": lambda c, u, h, q, b: c.patch(u, headers=h, json=b, params=q),
        }
        
        def api_function(**kwargs) -> Dict[str, Any]:
            """Dynamically generated API tool function."""
            try:
                # Handle case where a single 'kwargs' parameter might be passed as a JSON string
                # This can happen when LangChain/LLM serializes arguments incorrectly
                if len(kwargs) == 1 and 'kwargs' in kwargs:
                    kwargs_value = kwargs['kwargs']
                    if isinstance(kwargs_value, str):
                        try:
                            # Try to parse as JSON
                            parsed = json.loads(kwargs_value)
                            if isinstance(parsed, dict):
                                kwargs = parsed
                            else:
                                # If parsed value is not a dict, keep original kwargs but log warning
                                logger.warning(f"Tool {tool_name} received kwargs as string but parsed value is not a dict: {parsed}")
                                kwargs = {}
                        except (json.JSONDecodeError, ValueError) as e:
                            # If it's an empty string or invalid JSON, treat as empty kwargs
                            if kwargs_value.strip() in ['{}', '']:
                                kwargs = {}
                            else:
                                logger.warning(f"Tool {tool_name} received invalid JSON string for kwargs: {kwargs_value}, error: {e}")
                                kwargs = {}
                
                # Separate params
                path_values = {k: v for k, v in kwargs.items() if k in path_params}
                query_values = {k: v for k, v in kwargs.items() if k in api_spec.query_params}
                body_values = {k: v for k, v in kwargs.items() 
                             if k not in path_params and k not in api_spec.query_params and v is not None}
                
                # Validate required path parameters
                missing_path_params = [p for p in path_params if p not in path_values or path_values[p] is None]
                if missing_path_params:
                    return {
                        "error": f"Missing required path parameters: {', '.join(missing_path_params)}",
                        "status": "validation_error",
                        "missing_params": missing_path_params
                    }
                
                # Build URL
                url = api_spec.url
                
                # Replace {{base_url}} placeholder with actual base URL at runtime
                if "{{base_url}}" in url:
                    base_url = BANKING_API_BASE_URL or ""
                    if not base_url:
                        return {
                            "error": "BANKING_API_BASE_URL not configured. Please set it in environment variables.",
                            "status": "config_error"
                        }
                    url = url.replace("{{base_url}}", base_url.rstrip("/"))
                
                # Replace path parameters (validate all are replaced)
                for param, value in path_values.items():
                    if value is None:
                        return {
                            "error": f"Path parameter '{param}' cannot be None",
                            "status": "validation_error"
                        }
                    url = url.replace(f"{{{param}}}", str(value))
                
                # Check if any path parameters remain unreplaced
                remaining_params = re.findall(r'\{(\w+)\}', url)
                if remaining_params:
                    return {
                        "error": f"Unreplaced path parameters in URL: {', '.join(remaining_params)}",
                        "status": "validation_error",
                        "unreplaced_params": remaining_params
                    }
                
                # Build headers
                headers = dict(api_spec.headers)
                
                # Handle authentication
                if use_oauth2_auth:
                    # Get OAuth2 access token
                    from src.core.utils.oauth2_auth import get_oauth2_access_token
                    oauth2_token = get_oauth2_access_token()
                    if oauth2_token:
                        headers["Authorization"] = f"Bearer {oauth2_token}"
                    else:
                        headers["Authorization"] = f"Bearer A21AALMiDcIh5KMbcvqHU0X1g0hhQzhfEOIeyKFxWhvbaUeDbRWIqcYJqiSCX3guH1Hys_Y4yeBLcaEo8VMT6MeSfx8klCeDA"
                elif auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                
                # Make request
                method = api_spec.method.upper()
                if method not in method_map:
                    return {"error": f"Unsupported method: {method}", "status": "error"}
                
                # Filter out None values from query and body
                query_values = {k: v for k, v in query_values.items() if v is not None}
                body_values = {k: v for k, v in body_values.items() if v is not None}
                
                with httpx.Client(timeout=timeout) as client:
                    response = method_map[method](client, url, headers, query_values, body_values)
                    try:
                        data = response.json()
                    except Exception:
                        data = {"text": response.text, "raw_response": response.text[:1000]}
                    
                    if 200 <= response.status_code < 300:
                        
                        return {
                            "data": data, 
                            "status_code": response.status_code, 
                            "status": "success",
                            "headers": dict(response.headers)
                        }
                    
                    # Enhanced error handling
                    error_message = "API request failed"
                    error_details = {}
                    
                    if isinstance(data, dict):
                        error_message = data.get("error", data.get("error_description", data.get("message", error_message)))
                        error_details = {
                            "error_code": data.get("error_code"),
                            "error_description": data.get("error_description"),
                            "details": data.get("details"),
                            "links": data.get("links")
                        }
                    elif isinstance(data, str):
                        error_message = data
                    
                    return {
                        "error": error_message,
                        "status_code": response.status_code,
                        "data": data,
                        "error_details": error_details,
                        "status": "error"
                    }
            except httpx.TimeoutException:
                return {"error": f"Request timeout after {timeout}s", "status": "timeout"}
            except Exception as e:
                logger.error(f"Error in API tool {tool_name}: {e}", exc_info=True)
                return {"error": f"Request failed: {str(e)}", "status": "error"}
        
        api_function.__name__ = tool_name
        api_function.__doc__ = description
        return api_function
    
    def _build_args_schema(self, request_schema: Dict[str, Any], path_params: List[str] = None) -> Optional[type]:
        """Build Pydantic model for tool arguments.
        
        Args:
            request_schema: Request body/query parameter schema
            path_params: List of path parameters from URL (e.g., ['order_id', 'account_id'])
        """
        type_map = {
            "number": float, "float": float,
            "integer": int, "int": int,
            "boolean": bool, "bool": bool,
            "array": list, "object": dict
        }
        
        fields = {}
        
        # Add path parameters to schema (these are always required)
        if path_params:
            for param_name in path_params:
                # Generate a helpful description based on parameter name
                # Common patterns: order_id -> "The order ID", account_id -> "The account ID", etc.
                param_display = param_name.replace('_', ' ').title()
                if param_name.endswith('_id'):
                    entity = param_name[:-3].replace('_', ' ')  # Remove '_id' suffix
                    desc = f"The {entity} ID (required path parameter)"
                else:
                    desc = f"The {param_display.lower()} (required path parameter)"
                # Path parameters are always required
                fields[param_name] = (str, Field(..., description=desc))
        
        # Add request schema parameters
        if request_schema:
            for param_name, param_spec in request_schema.items():
                # Skip if already added as path parameter
                if param_name in fields:
                    continue
                    
                if not isinstance(param_spec, dict):
                    fields[param_name] = (Optional[str], Field(default=None))
                    continue
                
                param_type = type_map.get(param_spec.get("type", "string"), str)
                required = param_spec.get("required", False)
                default = param_spec.get("default")
                desc = param_spec.get("description", f"Parameter {param_name}")
                
                if required and default is None:
                    fields[param_name] = (param_type, Field(..., description=desc))
                else:
                    fields[param_name] = (Optional[param_type], Field(default=default, description=desc))
        
        return create_model("ArgsSchema", **fields) if fields else None
    
    def register_api_endpoint(
        self,
        domain: str,
        tool_name: str,
        description: str,
        api_spec: Dict[str, Any],
        auth_token: Optional[str] = None,
        timeout: int = 30,
        use_oauth2_auth: bool = False,
        skip_rate_limit: bool = False
    ) -> BaseTool:
        """
        Register an API endpoint as a LangChain tool.
        
        Args:
            domain: Domain name (e.g., "accounts", "payments")
            tool_name: Tool name (e.g., "accounts.get_balance")
            description: Tool description
            api_spec: API spec dict with method, url, headers, request_schema, query_params
            auth_token: Optional auth token
            timeout: Request timeout in seconds
            use_oauth2_auth: Use OAuth2 authentication
            skip_rate_limit: Skip rate limiting (for bulk operations)
            
        Returns:
            Registered LangChain tool
            
        Raises:
            ValueError: If rate limit exceeded
        """
        # Rate limiting check (skip for bulk operations)
        if not skip_rate_limit:
            if not check_rate_limit(
                "tool_registry",
                RATE_LIMIT_TOOL_REGISTRY_CAPACITY,
                RATE_LIMIT_TOOL_REGISTRY_REFILL
            ):
                raise ValueError(
                    f"Rate limit exceeded for tool registry. "
                    f"Please wait before registering more tools."
                )
        
        # Create API spec
        spec = APISpec(**api_spec)
        
        # Extract path parameters from URL (before creating function)
        # Replace {{base_url}} temporarily to avoid matching {base_url} inside it
        temp_url = spec.url.replace("{{base_url}}", "__BASE_URL_PLACEHOLDER__")
        path_params = re.findall(r'\{(\w+)\}', temp_url)
        
        # Create API function
        api_function = self._create_api_function(
            tool_name, description, spec, auth_token, timeout, use_oauth2_auth
        )
        
        # Build args schema (include path parameters)
        args_schema = self._build_args_schema(spec.request_schema, path_params=path_params)
        
        # Create a custom tool class that handles string inputs for kwargs
        class BankingAPITool(StructuredTool):
            """Custom tool that handles string inputs for kwargs parameter."""
            
            def _parse_input(
                self, tool_input: Union[str, dict], tool_call_id: Optional[str] = None
            ) -> Union[str, dict]:
                """Parse tool input, handling string JSON inputs for kwargs."""
                # If input is a dict, check if it has a 'kwargs' key with string value
                if isinstance(tool_input, dict):
                    if 'kwargs' in tool_input and isinstance(tool_input['kwargs'], str):
                        try:
                            # Try to parse the string as JSON
                            parsed = json.loads(tool_input['kwargs'])
                            if isinstance(parsed, dict):
                                # Replace kwargs string with parsed dict
                                tool_input = {k: v for k, v in tool_input.items() if k != 'kwargs'}
                                tool_input.update(parsed)
                            elif tool_input['kwargs'].strip() in ['{}', '']:
                                # Empty kwargs string, remove it
                                tool_input = {k: v for k, v in tool_input.items() if k != 'kwargs'}
                        except (json.JSONDecodeError, ValueError):
                            # If parsing fails and it's empty, just remove it
                            if tool_input['kwargs'].strip() in ['{}', '']:
                                tool_input = {k: v for k, v in tool_input.items() if k != 'kwargs'}
                
                # Call parent's _parse_input
                return super()._parse_input(tool_input, tool_call_id)
        
        # Create LangChain tool using custom BankingAPITool
        langchain_tool = BankingAPITool.from_function(
            func=api_function,
            name=tool_name,
            description=description,
            args_schema=args_schema
        )
        
        # Register tool
        tool_type = "banking_api"
        metadata = {
            "name": tool_name,
            "description": description,
            "domain": domain,
            "type": tool_type,
            "api_spec": api_spec,
            "use_oauth2_auth": use_oauth2_auth,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        self.register_tool(domain, langchain_tool, metadata, skip_rate_limit=skip_rate_limit)
        logger.info(f"Registered API endpoint: {tool_name} in domain '{domain}' (type: {tool_type})")
        return langchain_tool
    
    def register_tool(
        self,
        domain: str,
        tool: BaseTool,
        metadata: Optional[Dict[str, Any]] = None,
        save_to_file: bool = True,
        skip_rate_limit: bool = False
    ):
        """
        Register a tool for a domain.
        
        Args:
            domain: Domain name
            tool: LangChain tool to register
            metadata: Tool metadata
            save_to_file: Whether to save to file
            skip_rate_limit: Skip rate limiting (for bulk operations)
        
        Raises:
            ValueError: If rate limit exceeded
        """
        # Rate limiting check (only for external registrations, not internal loads)
        if save_to_file and not skip_rate_limit:
            if not check_rate_limit(
                "tool_registry",
                RATE_LIMIT_TOOL_REGISTRY_CAPACITY,
                RATE_LIMIT_TOOL_REGISTRY_REFILL
            ):
                raise ValueError(
                    f"Rate limit exceeded for tool registry. "
                    f"Please wait before registering more tools."
                )
        if domain not in self._domain_tools:
            self._domain_tools[domain] = []
        
        self._domain_tools[domain].append(tool)
        
        tool_name = metadata.get("name", tool.name) if metadata else tool.name
        self._tool_metadata[tool_name] = metadata or {}
        
        # Auto-save banking API tools
        if save_to_file and metadata and metadata.get("type") == "banking_api":
            self._save_to_file()
    
    def _save_to_file(self):
        """Save registered banking API tools to JSON file (atomic write)."""
        try:
            # Save banking API tools
            api_tools = {
                name: {
                    "domain": meta.get("domain"),
                    "metadata": meta
                }
                for name, meta in self._tool_metadata.items()
                if meta.get("type") == "banking_api"
            }
            
            data = {
                "version": "1.0",
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "total_tools": len(api_tools),
                "tools": api_tools
            }
            
            # Atomic file write to prevent corruption
            temp_file = f"{self._persistence_file}.tmp"
            try:
                # Write to temporary file first
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                # Atomic replace (works on Unix/Linux/Mac)
                # On Windows, this is still safer than direct write
                try:
                    os.replace(temp_file, self._persistence_file)
                except OSError:
                    # Fallback for Windows: delete old file, then rename
                    if os.path.exists(self._persistence_file):
                        os.remove(self._persistence_file)
                    os.rename(temp_file, self._persistence_file)
                
                logger.info(f"Saved {len(api_tools)} API tools to {self._persistence_file}")
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise e
        except Exception as e:
            logger.error(f"Failed to save tool registry: {e}", exc_info=True)
    
    def _load_from_file(self):
        """Load registered banking API tools from JSON file."""
        if not os.path.exists(self._persistence_file):
            return
        
        try:
            with open(self._persistence_file, 'r') as f:
                data = json.load(f)
            
            tools_data = data.get("tools", {})
            if not tools_data:
                return
            
            loaded_count = 0
            for tool_name, tool_info in tools_data.items():
                try:
                    metadata = tool_info.get("metadata", {})
                    domain = tool_info.get("domain") or metadata.get("domain")
                    api_spec = metadata.get("api_spec")
                    
                    if not domain or not api_spec:
                        continue
                    
                    # Recreate tool from API spec
                    spec = APISpec(**api_spec)
                    use_oauth2_auth = metadata.get("use_oauth2_auth", False)
                    
                    # Extract path parameters from URL
                    temp_url = spec.url.replace("{{base_url}}", "__BASE_URL_PLACEHOLDER__")
                    path_params = re.findall(r'\{(\w+)\}', temp_url)
                    
                    api_function = self._create_api_function(
                        tool_name,
                        metadata.get("description", ""),
                        spec,
                        auth_token=None,
                        timeout=30,
                        use_oauth2_auth=use_oauth2_auth
                    )
                    args_schema = self._build_args_schema(spec.request_schema, path_params=path_params)
                    langchain_tool = StructuredTool.from_function(
                        func=api_function,
                        name=tool_name,
                        description=metadata.get("description", ""),
                        args_schema=args_schema
                    )
                    
                    # Register without saving (to avoid recursion)
                    self.register_tool(domain, langchain_tool, metadata, save_to_file=False)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load tool {tool_name}: {e}")
            
            logger.info(f"Loaded {loaded_count} API tools from {self._persistence_file}")
        except Exception as e:
            logger.error(f"Failed to load tool registry: {e}", exc_info=True)
    
    def get_tools_for_domains(self, domains: List[str]) -> List[BaseTool]:
        """Get all tools for specified domains."""
        tools = []
        for domain in domains:
            if domain in self._domain_tools:
                tools.extend(self._domain_tools[domain])
        
        # Always include RAG tools
        if "rag" in self._domain_tools:
            for rag_tool in self._domain_tools["rag"]:
                if rag_tool not in tools:
                    tools.append(rag_tool)
        
        return tools
    
    def unregister_tool(self, tool_name: str, save_to_file: bool = True) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            tool_name: Name of the tool to unregister
            save_to_file: Whether to persist the change to file
            
        Returns:
            True if tool was found and unregistered, False otherwise
        """
        # Find and remove tool from domain
        tool_found = False
        for domain, tools in self._domain_tools.items():
            tools_to_remove = [tool for tool in tools if tool.name == tool_name]
            if tools_to_remove:
                for tool in tools_to_remove:
                    tools.remove(tool)
                tool_found = True
                logger.info(f"Removed tool '{tool_name}' from domain '{domain}'")
                break
        
        # Remove metadata
        if tool_name in self._tool_metadata:
            del self._tool_metadata[tool_name]
            tool_found = True
        
        # Save to file if requested
        if tool_found and save_to_file:
            self._save_to_file()
        
        return tool_found
    
    def get_tools_for_domain(self, domain: str) -> List[BaseTool]:
        """Get tools for a single domain."""
        return self._domain_tools.get(domain, [])
    
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self._domain_tools.keys())
    
    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get metadata for a specific tool."""
        return self._tool_metadata.get(tool_name, {})
    
    def get_total_tools_count(self) -> int:
        """Get total number of registered tools."""
        return sum(len(tools) for tools in self._domain_tools.values())
    
    def get_tools_count(self) -> Dict[str, int]:
        """Get count of tools per domain."""
        return {
            domain: len(tools)
            for domain, tools in self._domain_tools.items()
        }
    
    def list_tools_by_domain(self) -> Dict[str, List[str]]:
        """List all tools organized by domain."""
        return {
            domain: [tool.name for tool in tools]
            for domain, tools in self._domain_tools.items()
        }


# Global tool registry instance
tool_registry = ToolRegistry()
