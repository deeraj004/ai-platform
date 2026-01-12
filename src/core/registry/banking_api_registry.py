"""
Banking API Registry - Simplified wrapper for registering banking APIs.
"""
from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool
from settings import logger, BANKING_DOMAINS
from src.core.registry.tool_registry import tool_registry, APISpec


class BankingAPIRegistry:
    """Simplified banking API registry wrapper."""
    
    def __init__(self):
        self.registered_apis: Dict[str, List[str]] = {}
    
    def register_api_from_config(
        self, domain: str, api_name: str, api_tool: BaseTool,
        description: str, category: str = "api"
    ):
        """Register a single banking API with pre-built tool."""
        if domain not in BANKING_DOMAINS:
            logger.warning(f"Domain '{domain}' not in recognized banking domains")
        
        tool_registry.register_tool(domain, api_tool, {
            "name": api_name,
            "description": description,
            "domain": domain,
            "category": category,
            "type": "banking_api"
        })
        
        self.registered_apis.setdefault(domain, []).append(api_name)
        logger.debug(f"Registered API: {api_name} in domain '{domain}'")
    
    def register_api_from_endpoint(
        self,
        domain: str,
        tool_name: str,
        description: str,
        api_spec: Dict[str, Any],
        auth_token: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Register a banking API from endpoint specification."""
        if domain not in BANKING_DOMAINS:
            logger.warning(f"Domain '{domain}' not in recognized banking domains")
        
        tool = tool_registry.register_api_endpoint(
            domain, tool_name, description, api_spec, auth_token, timeout
        )
        
        self.registered_apis.setdefault(domain, []).append(tool_name)
        
        return {
            "status": "success",
            "tool_name": tool_name,
            "domain": domain,
            "message": "Tool registered successfully"
        }
    
    def register_apis_from_list(self, api_list: List[Dict[str, Any]]):
        """Register multiple banking APIs from a list."""
        for api_config in api_list:
            if api_config.get("api_tool"):
                # Pre-built tool
                self.register_api_from_config(
                    api_config.get("domain"),
                    api_config.get("api_name"),
                    api_config.get("api_tool"),
                    api_config.get("description", ""),
                    api_config.get("category", "api")
                )
            elif api_config.get("api_spec"):
                # API endpoint
                self.register_api_from_endpoint(
                    api_config.get("domain"),
                    api_config.get("api_name"),
                    api_config.get("description", ""),
                    api_config.get("api_spec"),
                    api_config.get("auth_token"),
                    api_config.get("timeout", 30)
                )
        
        logger.info(f"Registered {len(api_list)} banking APIs")
    
    def get_registered_apis_summary(self) -> Dict[str, Any]:
        """Get summary of registered banking APIs."""
        return {
            "total_apis": tool_registry.get_total_tools_count(),
            "domains": {domain: len(tool_registry.get_tools_for_domain(domain)) 
                       for domain in tool_registry.list_domains()},
            "registered_by_domain": self.registered_apis
        }


# Global banking API registry instance
banking_api_registry = BankingAPIRegistry()
