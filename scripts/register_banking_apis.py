#!/usr/bin/env python3
"""
Bulk Register Banking APIs from Postman Collection
Generic script to parse any banking API Postman collection and register all APIs as tools.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils.postman_parser import parse_postman_collection
from src.core.registry.tool_registry import tool_registry
from settings import logger


def register_banking_apis(collection_path: str, clear_existing: bool = True, use_oauth2_auth: bool = True):
    """
    Register all banking APIs from Postman collection.
    
    Args:
        collection_path: Path to Postman collection JSON file
        clear_existing: If True, clear existing banking API tools before registering
        use_oauth2_auth: If True, use OAuth2 authentication for API calls
    """
    logger.info("Starting banking API registration from Postman collection")
    
    # Reset rate limiter for bulk operations (clear any existing state)
    try:
        from src.core.utils.rate_limiter import _rate_limiter
        # Clear the tool_registry limiter if it exists
        if hasattr(_rate_limiter, '_limiters') and 'tool_registry' in _rate_limiter._limiters:
            del _rate_limiter._limiters['tool_registry']
            logger.info("Reset rate limiter for bulk registration")
    except Exception as e:
        logger.debug(f"Could not reset rate limiter: {e}")
    
    # Parse Postman collection
    endpoints = parse_postman_collection(collection_path)
    
    if not endpoints:
        logger.error("No endpoints found in Postman collection")
        return
    
    logger.info(f"Found {len(endpoints)} API endpoints to register")
    
    # Clear existing banking API tools if requested
    if clear_existing:
        logger.info("Clearing existing banking API tools...")
        banking_tools = [
            name for name in tool_registry._tool_metadata.keys()
            if tool_registry._tool_metadata[name].get("type") in ["banking_api", "api"]
        ]
        for tool_name in banking_tools:
            tool_registry.unregister_tool(tool_name, save_to_file=False)
        logger.info(f"Cleared {len(banking_tools)} existing banking API tools")
    
    # Register each endpoint
    registered_count = 0
    failed_count = 0
    
    for endpoint in endpoints:
        try:
            # Build API spec
            api_spec = {
                "method": endpoint["method"],
                "url": endpoint["url"],
                "headers": endpoint.get("headers", {}),
                "request_schema": endpoint.get("request_schema", {}),
                "query_params": endpoint.get("query_params", [])
            }
            
            # Register with OAuth2 authentication if enabled
            # Skip rate limiting for bulk registration
            tool_registry.register_api_endpoint(
                domain=endpoint["domain"],
                tool_name=endpoint["tool_name"],
                description=endpoint.get("description", endpoint["name"]),
                api_spec=api_spec,
                use_oauth2_auth=use_oauth2_auth,
                timeout=30,
                skip_rate_limit=True  # Skip rate limiting for bulk operations
            )
            
            registered_count += 1
            
            if registered_count % 50 == 0:
                logger.info(f"Registered {registered_count}/{len(endpoints)} endpoints...")
        
        except Exception as e:
            failed_count += 1
            logger.warning(f"Failed to register {endpoint.get('tool_name', 'unknown')}: {e}")
            # Only stop if we have too many consecutive failures (not rate limit related)
            if failed_count > 50:  # Increased threshold since rate limiting is skipped
                logger.error("Too many failures, stopping registration")
                break
    
    logger.info(f"Banking API registration complete: {registered_count} registered, {failed_count} failed")
    
    # Save to file
    try:
        tool_registry._save_to_file()
        logger.info("Tool registry saved to file")
    except Exception as e:
        logger.error(f"Failed to save tool registry: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register banking APIs from Postman collection")
    parser.add_argument("collection_path", nargs="?", help="Path to Postman collection JSON file")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing tools")
    parser.add_argument("--no-oauth2", action="store_true", help="Don't use OAuth2 authentication")
    
    args = parser.parse_args()
    
    # Default collection path if not provided
    if not args.collection_path:
        # Try to find common Postman collection files
        possible_files = [
            project_root / "PayPal APIs.postman_collection.json",
            project_root / "Banking APIs.postman_collection.json",
            project_root / "APIs.postman_collection.json",
        ]
        
        collection_path = None
        for path in possible_files:
            if path.exists():
                collection_path = path
                break
        
        if not collection_path:
            logger.error("Postman collection not found. Please provide path as argument.")
            logger.info("Usage: python scripts/register_banking_apis.py <collection_path>")
            sys.exit(1)
    else:
        collection_path = Path(args.collection_path)
        if not collection_path.exists():
            logger.error(f"Postman collection not found at: {collection_path}")
            sys.exit(1)
    
    # Register all banking APIs
    register_banking_apis(
        str(collection_path),
        clear_existing=not args.no_clear,
        use_oauth2_auth=not args.no_oauth2
    )
    
    # Print summary
    total_tools = tool_registry.get_total_tools_count()
    logger.info(f"Total tools in registry: {total_tools}")
    
    # Print domain breakdown
    domain_counts = tool_registry.get_tools_count()
    logger.info("Tools by domain:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count}")

