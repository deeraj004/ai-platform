"""
System Search Tool - Allows agent to query system capabilities, tools, logs, and status.
This tool enables the agent to answer questions about its own capabilities.
"""
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from datetime import datetime
from settings import (
    logger,
    RATE_LIMIT_SYSTEM_SEARCH_CAPACITY,
    RATE_LIMIT_SYSTEM_SEARCH_REFILL
)
from src.core.utils.rate_limiter import check_rate_limit


class SystemSearchInput(BaseModel):
    query: str = Field(description="What to search for (e.g., 'invoice tools', 'payment APIs', 'system status')")
    search_type: str = Field(
        default="tools",
        description="Type of search: 'tools', 'logs', 'status', 'capabilities', 'metadata'"
    )
    domain_filter: Optional[str] = Field(default=None, description="Filter by domain (optional)")
    limit: int = Field(default=10, description="Maximum number of results to return")


def system_search_tool_impl(
    query: str,
    search_type: str = "tools",
    domain_filter: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search the system for information about tools, capabilities, logs, or status.
    This tool enables the agent to answer questions about its own capabilities.
    
    Examples:
    - "What tools are available for managing invoices?" -> search_type="tools", query="invoice"
    - "What can you help me with?" -> search_type="capabilities"
    - "What's the system status?" -> search_type="status"
    - "Get details about orders.create_order" -> search_type="metadata", query="orders.create_order"
    """
    # Rate limiting check
    if not check_rate_limit(
        "system_search",
        RATE_LIMIT_SYSTEM_SEARCH_CAPACITY,
        RATE_LIMIT_SYSTEM_SEARCH_REFILL
    ):
        return {
            "error": "Rate limit exceeded for system search. Please try again later.",
            "rate_limited": True,
            "search_type": search_type,
            "query": query
        }
    
    try:
        # Import tool_registry here to avoid circular imports
        from src.core.registry.tool_registry import tool_registry
        
        if search_type == "tools":
            # Search tool registry by name/description
            results = []
            for domain, tools in tool_registry.list_tools_by_domain().items():
                if domain_filter and domain != domain_filter:
                    continue
                for tool_name in tools:
                    metadata = tool_registry.get_tool_metadata(tool_name)
                    if not metadata:
                        continue
                    desc = metadata.get("description", "").lower()
                    tool_name_lower = tool_name.lower()
                    query_lower = query.lower()
                    
                    # Check if query matches tool name or description
                    if query_lower in tool_name_lower or query_lower in desc:
                        results.append({
                            "name": tool_name,
                            "domain": domain,
                            "description": metadata.get("description", ""),
                            "category": metadata.get("category", "api")
                        })
            
            return {
                "search_type": "tools",
                "query": query,
                "results": results[:limit],
                "total_found": len(results),
                "returned": min(len(results), limit)
            }
        
        elif search_type == "capabilities":
            # List all system capabilities
            domains = tool_registry.list_domains()
            tools_by_domain = tool_registry.list_tools_by_domain()
            tool_counts = tool_registry.get_tools_count()
            
            return {
                "search_type": "capabilities",
                "total_domains": len(domains),
                "total_tools": tool_registry.get_total_tools_count(),
                "domains": domains,
                "tools_by_domain": tools_by_domain,
                "tool_counts": tool_counts,
                "summary": f"System has {tool_registry.get_total_tools_count()} tools across {len(domains)} domains"
            }
        
        elif search_type == "status":
            # Get system status
            return {
                "search_type": "status",
                "system_status": "operational",
                "total_tools_registered": tool_registry.get_total_tools_count(),
                "domains_available": len(tool_registry.list_domains()),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        elif search_type == "metadata":
            # Get detailed metadata for specific tool
            metadata = tool_registry.get_tool_metadata(query)
            if metadata:
                return {
                    "search_type": "metadata",
                    "tool_name": query,
                    "metadata": metadata,
                    "found": True
                }
            return {
                "search_type": "metadata",
                "tool_name": query,
                "found": False,
                "error": "Tool not found"
            }
        
        else:
            return {
                "error": f"Unknown search_type: {search_type}",
                "valid_types": ["tools", "capabilities", "status", "metadata"],
                "search_type": search_type
            }
    
    except Exception as e:
        logger.error(f"System search error: {e}", exc_info=True)
        return {
            "error": str(e),
            "search_type": search_type,
            "query": query
        }


# Create LangChain tool
system_search_tool = StructuredTool.from_function(
    func=system_search_tool_impl,
    name="system.search",
    description="Search system capabilities, tools, logs, or status. Use this when users ask about what tools are available, system capabilities, or system status. Examples: 'What tools do you have for invoices?', 'What can you help me with?', 'What's the system status?'",
    args_schema=SystemSearchInput
)

