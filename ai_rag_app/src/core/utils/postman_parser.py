"""
Postman Collection Parser
Generic parser for Postman collection JSON files to extract API endpoints for tool registration.
Works with any banking API Postman collection.
"""
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from settings import logger


def extract_path_params(url: str) -> List[str]:
    """Extract path parameters from URL (e.g., {order_id} or :order_id)."""
    # Handle both {param} and :param formats
    params = re.findall(r'\{(\w+)\}', url)
    params.extend(re.findall(r':(\w+)', url))
    return list(set(params))  # Remove duplicates


def normalize_url(url: str, base_url_var: str = "{{base_url}}") -> str:
    """
    Normalize URL by replacing variables with placeholders.
    Keeps {{base_url}} as placeholder - will be replaced at runtime.
    """
    # Convert :param to {param} format for consistency
    url = re.sub(r':(\w+)', r'{\1}', url)
    # Keep base_url as variable - will be resolved from settings
    return url


def extract_request_body_schema(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract request body schema from Postman body."""
    schema = {}
    
    if not body:
        return schema
    
    mode = body.get("mode", "")
    
    if mode == "raw" and body.get("raw"):
        try:
            # Try to parse JSON
            raw_data = body.get("raw", "")
            if raw_data.strip().startswith("{"):
                json_data = json.loads(raw_data)
                # Convert JSON to schema format
                schema = _json_to_schema(json_data)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not parse raw body as JSON: {e}")
    
    elif mode == "urlencoded" and body.get("urlencoded"):
        # Form-encoded data
        for item in body.get("urlencoded", []):
            if not item.get("disabled", False):
                key = item.get("key", "")
                desc = item.get("description", "")
                schema[key] = {
                    "type": "string",
                    "required": not item.get("disabled", False),
                    "description": desc or f"Parameter {key}"
                }
    
    elif mode == "formdata" and body.get("formdata"):
        # Form data
        for item in body.get("formdata", []):
            if not item.get("disabled", False):
                key = item.get("key", "")
                desc = item.get("description", "")
                schema[key] = {
                    "type": "string",
                    "required": not item.get("disabled", False),
                    "description": desc or f"Parameter {key}"
                }
    
    return schema


def _json_to_schema(json_data: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively convert JSON object to schema format."""
    schema = {}
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                schema[key] = {
                    "type": "object",
                    "required": True,
                    "description": f"Object parameter {key}",
                    "properties": _json_to_schema(value, full_key)
                }
            elif isinstance(value, list):
                schema[key] = {
                    "type": "array",
                    "required": True,
                    "description": f"Array parameter {key}"
                }
            else:
                schema[key] = {
                    "type": _get_type_from_value(value),
                    "required": True,
                    "description": f"Parameter {key}"
                }
    elif isinstance(json_data, list) and json_data:
        # Handle array of objects
        if isinstance(json_data[0], dict):
            return _json_to_schema(json_data[0], prefix)
    
    return schema


def _get_type_from_value(value: Any) -> str:
    """Get schema type from Python value."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    else:
        return "string"


def extract_query_params(url_obj: Dict[str, Any]) -> List[str]:
    """Extract query parameters from Postman URL object."""
    query_params = []
    
    if isinstance(url_obj, dict):
        if "query" in url_obj:
            for param in url_obj.get("query", []):
                if not param.get("disabled", False):
                    query_params.append(param.get("key", ""))
        elif "raw" in url_obj:
            # Parse raw URL
            raw_url = url_obj["raw"]
            if "?" in raw_url:
                query_string = raw_url.split("?")[1].split("#")[0]
                for param in query_string.split("&"):
                    if "=" in param:
                        query_params.append(param.split("=")[0])
    
    return query_params


def parse_postman_item(item: Dict[str, Any], parent_folder: str = "") -> List[Dict[str, Any]]:
    """
    Parse a Postman collection item (folder or request).
    
    Returns:
        List of parsed API endpoint dictionaries
    """
    endpoints = []
    
    # Check if this is a folder (has "item" key)
    if "item" in item:
        # This is a folder
        folder_name = item.get("name", "Unknown")
        current_folder = f"{parent_folder}.{folder_name}" if parent_folder else folder_name
        
        # Recursively parse items in folder
        for sub_item in item.get("item", []):
            endpoints.extend(parse_postman_item(sub_item, current_folder))
    
    # Check if this is a request (has "request" key)
    elif "request" in item:
        endpoint = parse_postman_request(item, parent_folder)
        if endpoint:
            endpoints.append(endpoint)
    
    return endpoints


def parse_postman_request(item: Dict[str, Any], folder_path: str = "") -> Optional[Dict[str, Any]]:
    """
    Parse a single Postman request into our API spec format.
    
    Returns:
        Dictionary with API specification or None if invalid
    """
    try:
        name = item.get("name", "Unnamed")
        request = item.get("request", {})
        
        if not request:
            return None
        
        # Extract method
        method = request.get("method", "GET").upper()
        
        # Extract URL
        url_obj = request.get("url", {})
        if isinstance(url_obj, str):
            url = url_obj
        elif isinstance(url_obj, dict):
            # Build URL from parts
            if "raw" in url_obj:
                url = url_obj["raw"]
            else:
                # Build from host, path, etc.
                host = url_obj.get("host", ["{{base_url}}"])
                path = url_obj.get("path", [])
                if isinstance(host, list):
                    host = host[0] if host else "{{base_url}}"
                url = f"{host}/{'/'.join(path)}" if path else host
        
        # Normalize URL
        normalized_url = normalize_url(url)
        
        # Extract path parameters
        path_params = extract_path_params(normalized_url)
        
        # Extract query parameters
        query_params = extract_query_params(url_obj)
        
        # Extract request body schema
        body = request.get("body", {})
        request_schema = extract_request_body_schema(body)
        
        # Extract headers (excluding auth headers)
        headers = {}
        for header in request.get("header", []):
            if not header.get("disabled", False):
                key = header.get("key", "")
                value = header.get("value", "")
                # Skip auth headers (will be added by auth handler)
                if key.lower() not in ["authorization"]:
                    headers[key] = value
        
        # Extract description
        description = item.get("request", {}).get("description", "")
        if not description:
            description = name
        
        # Determine domain from folder path
        domain_parts = folder_path.split(".") if folder_path else []
        if domain_parts:
            domain = domain_parts[0].lower().replace(" ", "_")
        else:
            domain = "api"
        
        # Generate tool name
        tool_name_parts = [domain]
        if len(domain_parts) > 1:
            tool_name_parts.extend([p.lower().replace(" ", "_") for p in domain_parts[1:]])
        tool_name_parts.append(name.lower().replace(" ", "_").replace("-", "_"))
        tool_name = ".".join(tool_name_parts)
        
        # Clean up tool name
        tool_name = re.sub(r'[^a-z0-9._]', '', tool_name)
        tool_name = re.sub(r'_+', '_', tool_name)
        tool_name = tool_name.strip('._')
        
        return {
            "tool_name": tool_name,
            "domain": domain,
            "name": name,
            "description": description,
            "method": method,
            "url": normalized_url,
            "path_params": path_params,
            "query_params": query_params,
            "request_schema": request_schema,
            "headers": headers
        }
    
    except Exception as e:
        logger.warning(f"Error parsing Postman request '{item.get('name', 'Unknown')}': {e}")
        return None


def parse_postman_collection(collection_path: str) -> List[Dict[str, Any]]:
    """
    Parse Postman collection JSON file.
    
    Args:
        collection_path: Path to Postman collection JSON file
        
    Returns:
        List of parsed API endpoint dictionaries
    """
    try:
        with open(collection_path, 'r', encoding='utf-8') as f:
            collection = json.load(f)
        
        endpoints = []
        
        # Parse all items in collection
        for item in collection.get("item", []):
            endpoints.extend(parse_postman_item(item))
        
        logger.info(f"Parsed {len(endpoints)} API endpoints from Postman collection")
        return endpoints
    
    except Exception as e:
        logger.error(f"Error parsing Postman collection: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # Test parser
    import sys
    if len(sys.argv) > 1:
        collection_path = sys.argv[1]
    else:
        collection_path = Path(__file__).parent.parent.parent.parent / "Banking APIs.postman_collection.json"
    
    endpoints = parse_postman_collection(str(collection_path))
    print(f"Parsed {len(endpoints)} endpoints")
    if endpoints:
        print(f"Sample endpoint: {endpoints[0]}")

