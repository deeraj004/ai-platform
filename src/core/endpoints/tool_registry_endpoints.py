"""
Tool Registry API Endpoints - REST API for registering and managing banking API tools.
Enhanced with strict input validation for security and data integrity.
"""
import re
import ipaddress
import secrets
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, status, Query, Path, Depends, Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, field_validator, model_validator, constr, conint

from src.core.registry.banking_api_registry import banking_api_registry
from src.core.registry.tool_registry import tool_registry, APISpec
from settings import logger, BANKING_DOMAINS, ENABLE_API_AUTH, API_USERNAME, API_PASSWORD

tool_registry_router = APIRouter(tags=["Tool Registry"])

# Basic Auth Security
security = HTTPBasic()


def verify_api_credentials(credentials: HTTPBasicCredentials = Security(security)) -> str:
    """
    Verify API credentials for protected endpoints.
    
    Args:
        credentials: HTTP Basic Auth credentials
    
    Returns:
        Username if credentials are valid
    
    Raises:
        HTTPException: If credentials are invalid
    """
    if not ENABLE_API_AUTH:
        return credentials.username
    
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    
    if not (correct_username and correct_password):
        logger.warning(f"Failed API authentication attempt from user: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    logger.debug(f"Successful API authentication for user: {credentials.username}")
    return credentials.username

# Constants for validation
ALLOWED_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
ALLOWED_AUTH_TYPES = {"bearer", "api_key", "oauth2", "none"}
ALLOWED_TOKEN_SOURCES = {"header", "env_var", "query_param"}
ALLOWED_RETRY_BACKOFF = {"exponential", "linear", "fixed"}
ALLOWED_CONTEXT_PRIORITY = {"high", "medium", "low"}
ALLOWED_RISK_LEVELS = {"read_only", "write", "financial", "admin"}
MAX_STRING_LENGTH = 1000
MAX_DESCRIPTION_LENGTH = 500

def truncate_description(description: str, max_length: int = MAX_DESCRIPTION_LENGTH) -> str:
    """
    Truncate description to max_length, removing HTML tags if present.
    
    Args:
        description: Description string (may contain HTML)
        max_length: Maximum length (default: MAX_DESCRIPTION_LENGTH)
    
    Returns:
        Truncated description
    """
    if not description:
        return ""
    
    # Remove HTML tags using regex
    clean_desc = re.sub(r'<[^>]+>', '', description)
    
    # Strip whitespace
    clean_desc = clean_desc.strip()
    
    # Truncate if too long
    if len(clean_desc) > max_length:
        clean_desc = clean_desc[:max_length - 3] + "..."
    
    return clean_desc
MAX_API_NAME_LENGTH = 100
MAX_URL_LENGTH = 2048
MAX_TOOLS_BULK = 100

# Regex patterns (API_NAME_PATTERN removed - allowing flexible naming with dots)
VERSION_PATTERN = r'^(\d+)\.(\d+)\.(\d+)(?:-[\w\.-]+)?(?:\+[\w\.-]+)?$'
TAG_PATTERN = r'^[a-zA-Z0-9_-]+$'
ENV_VAR_PATTERN = r'^[A-Z_][A-Z0-9_]*$'
HEADER_NAME_PATTERN = r'^[a-zA-Z0-9_-]+$'


# Request/Response Models with strict validation
class APISpecRequest(BaseModel):
    method: constr(
        strip_whitespace=True,
        to_upper=True,
        min_length=3,
        max_length=7
    ) = Field(
        ...,
        description="HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)"
    )
    
    url: constr(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_URL_LENGTH
    ) = Field(
        ...,
        description="API endpoint URL"
    )
    
    headers: Optional[Dict[
        constr(pattern=HEADER_NAME_PATTERN, max_length=100),
        constr(max_length=500)
    ]] = Field(
        default_factory=dict,
        description="HTTP headers as key-value pairs"
    )
    
    request_schema: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Request body schema definition"
    )
    
    response_schema: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Response body schema definition"
    )
    
    path_params: Optional[List[
        constr(pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$', max_length=50)
    ]] = Field(
        default_factory=list,
        description="List of path parameter names"
    )
    
    query_params: Optional[List[
        constr(pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$', max_length=50)
    ]] = Field(
        default_factory=list,
        description="List of query parameter names"
    )
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate HTTP method is allowed."""
        if v.upper() not in ALLOWED_HTTP_METHODS:
            raise ValueError(
                f"Invalid HTTP method '{v}'. Must be one of: {', '.join(ALLOWED_HTTP_METHODS)}"
            )
        return v.upper()
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format and prevent SSRF."""
        if not v:
            raise ValueError("URL cannot be empty")
        
        try:
            parsed = urlparse(v)
            
            # Must have scheme
            if not parsed.scheme:
                raise ValueError("URL must include a scheme (http:// or https://)")
            
            # Only HTTPS/HTTP allowed for security
            if parsed.scheme.lower() not in ('https', 'http'):
                raise ValueError("URL scheme must be 'http' or 'https'")
            
            # Must have hostname
            if not parsed.hostname:
                raise ValueError("URL must include a valid hostname")
            
            # Block localhost and internal IPs (SSRF protection)
            blocked_hosts = {
                'localhost', '127.0.0.1', '0.0.0.0', '::1',
                '169.254.169.254',  # AWS metadata
                '169.254.169.1',    # Azure metadata
            }
            
            if parsed.hostname.lower() in blocked_hosts:
                raise ValueError(f"URL hostname '{parsed.hostname}' is not allowed for security reasons")
            
            # Block private IP ranges
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError(f"Private/internal IP addresses are not allowed: {parsed.hostname}")
            except ValueError:
                pass  # Not an IP address, continue validation
            
            # Validate URL length
            if len(v) > MAX_URL_LENGTH:
                raise ValueError(f"URL exceeds maximum length of {MAX_URL_LENGTH} characters")
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid URL format: {str(e)}")
        
        return v
    
    @field_validator('headers')
    @classmethod
    def validate_headers(cls, v: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Validate headers dictionary."""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError("Headers must be a dictionary")
        
        if len(v) > 50:
            raise ValueError("Maximum 50 headers allowed")
        
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Header keys and values must be strings")
            if len(key) > 100:
                raise ValueError(f"Header key '{key}' exceeds maximum length of 100 characters")
            if len(value) > 500:
                raise ValueError(f"Header value for '{key}' exceeds maximum length of 500 characters")
        
        return v
    
    @field_validator('path_params', 'query_params')
    @classmethod
    def validate_param_list(cls, v: Optional[List[str]]) -> List[str]:
        """Validate parameter name lists."""
        if v is None:
            return []
        
        if not isinstance(v, list):
            raise ValueError("Must be a list of strings")
        
        if len(v) > 20:
            raise ValueError("Maximum 20 parameters allowed per list")
        
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate parameter names are not allowed")
        
        return v
    
    @model_validator(mode='after')
    def validate_path_params_in_url(self):
        """Ensure path_params are actually in the URL."""
        if self.path_params:
            url_params = re.findall(r'\{(\w+)\}', self.url)
            for param in self.path_params:
                if param not in url_params:
                    raise ValueError(
                        f"Path parameter '{param}' is not found in URL. "
                        f"URL should contain '{{{param}}}'"
                    )
        return self


class AuthConfigRequest(BaseModel):
    type: constr(
        strip_whitespace=True,
        to_lower=True,
        min_length=3,
        max_length=10
    ) = Field(
        default="none",
        description="Authentication type"
    )
    
    token_source: Optional[constr(
        strip_whitespace=True,
        to_lower=True,
        max_length=20
    )] = Field(
        default="header",
        description="Source of authentication token"
    )
    
    env_var_name: Optional[constr(
        strip_whitespace=True,
        pattern=ENV_VAR_PATTERN,
        max_length=100
    )] = Field(
        default=None,
        description="Environment variable name for token (if token_source is 'env_var')"
    )
    
    header_name: Optional[constr(
        strip_whitespace=True,
        pattern=HEADER_NAME_PATTERN,
        max_length=100
    )] = Field(
        default="Authorization",
        description="HTTP header name for authentication token"
    )
    
    api_key_name: Optional[constr(
        strip_whitespace=True,
        pattern=HEADER_NAME_PATTERN,
        max_length=100
    )] = Field(
        default="X-API-Key",
        description="Header or query parameter name for API key"
    )
    
    @field_validator('type')
    @classmethod
    def validate_auth_type(cls, v: str) -> str:
        """Validate authentication type."""
        if v.lower() not in ALLOWED_AUTH_TYPES:
            raise ValueError(
                f"Invalid auth type '{v}'. Must be one of: {', '.join(ALLOWED_AUTH_TYPES)}"
            )
        return v.lower()
    
    @field_validator('token_source')
    @classmethod
    def validate_token_source(cls, v: Optional[str]) -> Optional[str]:
        """Validate token source."""
        if v is None:
            return "header"
        if v.lower() not in ALLOWED_TOKEN_SOURCES:
            raise ValueError(
                f"Invalid token_source '{v}'. Must be one of: {', '.join(ALLOWED_TOKEN_SOURCES)}"
            )
        return v.lower()
    
    @model_validator(mode='after')
    def validate_env_var_required(self):
        """Ensure env_var_name is provided when token_source is 'env_var'."""
        if self.token_source == "env_var" and not self.env_var_name:
            raise ValueError("env_var_name is required when token_source is 'env_var'")
        return self


class ExecutionMetadataRequest(BaseModel):
    timeout: conint(ge=1, le=300) = Field(
        default=30,
        description="Request timeout in seconds (1-300)"
    )
    
    is_idempotent: bool = Field(
        default=True,
        description="Whether the operation is idempotent"
    )
    
    max_retries: conint(ge=0, le=10) = Field(
        default=3,
        description="Maximum number of retries (0-10)"
    )
    
    retry_backoff: constr(
        strip_whitespace=True,
        to_lower=True,
        max_length=15
    ) = Field(
        default="exponential",
        description="Retry backoff strategy"
    )
    
    requires_confirmation: bool = Field(
        default=False,
        description="Whether operation requires user confirmation"
    )
    
    is_transactional: bool = Field(
        default=False,
        description="Whether operation is transactional"
    )
    
    error_handling: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Error handling configuration"
    )
    
    @field_validator('retry_backoff')
    @classmethod
    def validate_retry_backoff(cls, v: str) -> str:
        """Validate retry backoff strategy."""
        if v.lower() not in ALLOWED_RETRY_BACKOFF:
            raise ValueError(
                f"Invalid retry_backoff '{v}'. Must be one of: {', '.join(ALLOWED_RETRY_BACKOFF)}"
            )
        return v.lower()
    
    @field_validator('error_handling')
    @classmethod
    def validate_error_handling(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate error handling configuration."""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError("error_handling must be a dictionary")
        
        if len(v) > 20:
            raise ValueError("error_handling dictionary cannot exceed 20 keys")
        
        return v


class CostMetadataRequest(BaseModel):
    estimated_tokens: conint(ge=1, le=100000) = Field(
        default=100,
        description="Estimated token count (1-100000)"
    )
    
    context_priority: constr(
        strip_whitespace=True,
        to_lower=True,
        max_length=10
    ) = Field(
        default="medium",
        description="Context priority level"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Whether caching is enabled"
    )
    
    cache_ttl: conint(ge=0, le=86400) = Field(
        default=3600,
        description="Cache TTL in seconds (0-86400, max 24 hours)"
    )
    
    @field_validator('context_priority')
    @classmethod
    def validate_context_priority(cls, v: str) -> str:
        """Validate context priority."""
        if v.lower() not in ALLOWED_CONTEXT_PRIORITY:
            raise ValueError(
                f"Invalid context_priority '{v}'. Must be one of: {', '.join(ALLOWED_CONTEXT_PRIORITY)}"
            )
        return v.lower()


class ToolRegistrationRequest(BaseModel):
    api_name: constr(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_API_NAME_LENGTH
    ) = Field(
        ...,
        description="Unique API tool name"
    )
    
    domain: constr(
        strip_whitespace=True,
        to_lower=True,
        min_length=1,
        max_length=50
    ) = Field(
        ...,
        description="Banking domain category"
    )
    
    description: constr(
        strip_whitespace=True,
        min_length=10,
        max_length=MAX_DESCRIPTION_LENGTH
    ) = Field(
        ...,
        description="Tool description (10-500 characters)"
    )
    
    category: constr(
        strip_whitespace=True,
        to_lower=True,
        max_length=50
    ) = Field(
        default="api",
        description="Tool category"
    )
    
    api_spec: APISpecRequest = Field(default=None,
        description="API specification"
    )
    
    auth_config: Optional[AuthConfigRequest] = Field(default=None,
        description="Authentication configuration"
    )
    
    execution_metadata: Optional[ExecutionMetadataRequest] = Field(default=None,
        description="Execution metadata"
    )
    
    
    @field_validator('domain')
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate domain is in allowed list."""
        if v.lower() not in BANKING_DOMAINS:
            raise ValueError(
                f"Invalid domain '{v}'. Must be one of: {', '.join(BANKING_DOMAINS)}"
            )
        return v.lower()

class ToolRegistrationResponse(BaseModel):
    status: constr(min_length=1, max_length=20)
    tool_name: constr(min_length=1, max_length=MAX_API_NAME_LENGTH)
    domain: constr(min_length=1, max_length=50)
    message: constr(max_length=500)
    metadata: Dict[str, Any]


class ToolInfoResponse(BaseModel):
    name: constr(min_length=1, max_length=MAX_API_NAME_LENGTH)
    domain: constr(min_length=1, max_length=50)
    description: constr(max_length=MAX_DESCRIPTION_LENGTH)
    category: constr(max_length=50)
    metadata: Dict[str, Any]


class BulkToolRegistrationRequest(BaseModel):
    tools: List[ToolRegistrationRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_TOOLS_BULK,
        description=f"List of tools to register (1-{MAX_TOOLS_BULK})"
    )
    
    @field_validator('tools')
    @classmethod
    def validate_tools_list(cls, v: List[ToolRegistrationRequest]) -> List[ToolRegistrationRequest]:
        """Validate bulk registration list."""
        if not v:
            raise ValueError("At least one tool must be provided")
        
        if len(v) > MAX_TOOLS_BULK:
            raise ValueError(f"Maximum {MAX_TOOLS_BULK} tools allowed per bulk registration")
        
        # Check for duplicate tool names
        tool_names = [tool.api_name for tool in v]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            raise ValueError(f"Duplicate tool names found: {', '.join(set(duplicates))}")
        
        return v


# Helper functions
def _convert_request_to_api_spec(request: ToolRegistrationRequest) -> Dict[str, Any]:
    """Convert request model to simplified API spec dict."""
    if not request.api_spec:
        raise ValueError("API specification is required")
    
    api_spec_dict = request.api_spec.model_dump()
    
    # Get auth token if configured
    auth_token = None
    if request.auth_config and request.auth_config.env_var_name:
        import os
        auth_token = os.getenv(request.auth_config.env_var_name)
    
    # Get timeout from execution metadata if provided
    timeout = 30
    if request.execution_metadata:
        timeout = request.execution_metadata.timeout
    
    return {
        "api_spec": api_spec_dict,
        "auth_token": auth_token,
        "timeout": timeout
    }


# Endpoints
@tool_registry_router.post("/tools/register", response_model=ToolRegistrationResponse, status_code=status.HTTP_201_CREATED)
async def register_tool(
    request: ToolRegistrationRequest,
    user: str = Depends(verify_api_credentials)
):
    """Register a banking API endpoint as a tool."""
    try:
        logger.info(f"Registering tool: {request.api_name} in domain {request.domain}")
        
        # Domain validation is now handled by Pydantic validator
        if tool_registry.get_tool_metadata(request.api_name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tool '{request.api_name}' already registered"
            )
        
        # Convert to simplified API spec
        spec_data = _convert_request_to_api_spec(request)
        result = banking_api_registry.register_api_from_endpoint(
            domain=request.domain,
            tool_name=request.api_name,
            description=request.description,
            api_spec=spec_data["api_spec"],
            auth_token=spec_data["auth_token"],
            timeout=spec_data["timeout"]
        )
        
        logger.info(f"Tool registered: {request.api_name}")
        return ToolRegistrationResponse(
            status=result["status"],
            tool_name=result["tool_name"],
            domain=result["domain"],
            message=result["message"],
            metadata={}
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering tool: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@tool_registry_router.get("/tools", response_model=List[ToolInfoResponse])
async def list_tools(
    domain: Optional[constr(to_lower=True, max_length=50)] = Query(None, description="Filter by domain"),
    category: Optional[constr(to_lower=True, max_length=50)] = Query(None, description="Filter by category"),
    search: Optional[constr(max_length=100)] = Query(None, description="Search term"),
    user: str = Depends(verify_api_credentials)
):
    """List all registered tools with optional filters."""
    try:
        tools = []
        search_lower = search.lower() if search else None
        
        for dom, tool_names in tool_registry.list_tools_by_domain().items():
            if domain and dom != domain:
                continue
            
            for tool_name in tool_names:
                metadata = tool_registry.get_tool_metadata(tool_name)
                if not metadata:
                    continue
                
                if category and metadata.get("category") != category:
                    continue
                
                if search_lower:
                    desc = metadata.get("description", "").lower()
                    if search_lower not in tool_name.lower() and search_lower not in desc:
                        continue
                
                tools.append(ToolInfoResponse(
                    name=tool_name,
                    domain=dom,
                    description=truncate_description(metadata.get("description", "")),
                    category=metadata.get("category", "api"),
                    metadata=metadata
                ))
        
        return tools
        
    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@tool_registry_router.get("/tools/{tool_name}", response_model=ToolInfoResponse)
async def get_tool(
    tool_name: str = Path(..., max_length=MAX_API_NAME_LENGTH, description="Tool name"),
    user: str = Depends(verify_api_credentials)
):
    """Get detailed information about a specific tool."""
    try:
        metadata = tool_registry.get_tool_metadata(tool_name)
        if not metadata:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found")
        
        # Find domain
        domain = next((dom for dom, names in tool_registry.list_tools_by_domain().items() 
                      if tool_name in names), "unknown")
        
        return ToolInfoResponse(
            name=tool_name,
            domain=domain,
            description=truncate_description(metadata.get("description", "")),
            category=metadata.get("category", "api"),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@tool_registry_router.delete("/tools/{tool_name}", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_tool(
    tool_name: str = Path(..., max_length=MAX_API_NAME_LENGTH, description="Tool name"),
    user: str = Depends(verify_api_credentials)
):
    """Unregister a tool."""
    try:
        if not tool_registry.get_tool_metadata(tool_name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found")
        
        logger.info(f"Unregister tool: {tool_name}")
        if not tool_registry.unregister_tool(tool_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found or already removed"
            )
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering tool: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@tool_registry_router.post("/tools/bulk-register", response_model=List[ToolRegistrationResponse])
async def bulk_register_tools(
    request: BulkToolRegistrationRequest,
    user: str = Depends(verify_api_credentials)
):
    """Register multiple tools at once."""
    results = []
    for tool_request in request.tools:
        try:
            result = await register_tool(tool_request)
            results.append(result)
        except HTTPException as e:
            logger.warning(f"Failed to register {tool_request.api_name}: {e.detail}")
            # Return error response for failed registration
            results.append(ToolRegistrationResponse(
                status="error",
                tool_name=tool_request.api_name,
                domain=tool_request.domain,
                message=f"Registration failed: {e.detail}",
                metadata={"error_code": e.status_code}
            ))
        except Exception as e:
            logger.error(f"Error registering {tool_request.api_name}: {e}", exc_info=True)
            # Return error response for unexpected errors
            results.append(ToolRegistrationResponse(
                status="error",
                tool_name=tool_request.api_name,
                domain=tool_request.domain,
                message=f"Unexpected error: {str(e)}",
                metadata={}
            ))
    
    return results
