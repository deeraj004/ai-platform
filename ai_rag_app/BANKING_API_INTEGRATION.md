# Banking API Integration Guide

This document explains how to register and use banking APIs from Postman collections in the Banking Agentic RAG System.

## Overview

The system supports generic banking APIs with OAuth2 authentication. Any banking API Postman collection can be parsed and registered as tools.

## Setup

### 1. Configure Banking API Credentials

Add the following to your `.env` file:

```bash
# Banking API OAuth2 Configuration
BANKING_API_CLIENT_ID=your_client_id
BANKING_API_CLIENT_SECRET=your_client_secret
BANKING_API_BASE_URL=https://api.bank.com  # Base URL for your banking API
BANKING_API_TOKEN_URL=https://api.bank.com/v1/oauth2/token  # OAuth2 token endpoint (optional, defaults to {BASE_URL}/v1/oauth2/token)
```

### 2. Register Banking APIs

Run the registration script to parse the Postman collection and register all banking APIs:

```bash
python scripts/register_banking_apis.py <path_to_postman_collection.json>
```

Or if the collection is in the project root with a common name:

```bash
python scripts/register_banking_apis.py
```

This will:
- Parse the Postman collection JSON file
- Extract all API endpoints
- Register them as tools with OAuth2 authentication
- Save to the tool registry

## Authentication

### OAuth2 Client Credentials Flow

The system automatically handles OAuth2 authentication:

1. **Token Generation**: Uses `client_credentials` grant type
2. **Token Management**: Automatically refreshes tokens before expiry
3. **Token Caching**: Thread-safe singleton pattern for efficient token management
4. **Automatic Injection**: Access tokens are automatically added to API requests

### Token Manager

The `OAuth2AuthManager` class handles:
- Automatic token refresh
- Thread-safe token access
- Token expiry management (refreshes 5 minutes before expiry)

## API Registration

### Tool Naming Convention

Banking APIs are registered with the following naming pattern:
```
{domain}.{subdomain}.{endpoint_name}
```

Example:
- `authorization.generate_access_token`
- `orders.create_order`
- `payments.capture_payment`

### Domain Organization

APIs are organized by domain based on the Postman collection folder structure:
- `authorization` - OAuth2 and authentication endpoints
- `orders` - Order management
- `payments` - Payment processing
- `accounts` - Account management
- etc.

## Usage

### Making API Calls

Once registered, banking APIs can be used like any other tool:

```python
# The system will automatically:
# 1. Get a valid OAuth2 access token
# 2. Add it to the Authorization header
# 3. Make the API call
# 4. Return the response

result = banking_tool.invoke({
    "order_id": "12345",
    # ... other parameters
})
```

### Error Handling

The system handles common API errors:
- **401 Unauthorized**: Token refresh is attempted automatically
- **403 Forbidden**: Returns error with details
- **Rate Limiting**: Returns appropriate error message

## Tool Registry

### Viewing Registered Tools

```python
from src.core.registry.tool_registry import tool_registry

# List all banking API tools
banking_tools = [
    name for name, meta in tool_registry._tool_metadata.items()
    if meta.get("type") == "banking_api"
]

print(f"Registered {len(banking_tools)} banking API tools")
```

### Tool Metadata

Each banking API tool includes:
- `name`: Tool name
- `description`: API description
- `domain`: Domain category
- `type`: "banking_api"
- `use_oauth2_auth`: True (indicates OAuth2 authentication)
- `api_spec`: Full API specification

## Troubleshooting

### Token Generation Fails

**Error**: "OAuth2 authentication failed"

**Solutions**:
1. Verify `BANKING_API_CLIENT_ID` and `BANKING_API_CLIENT_SECRET` are set correctly
2. Check that credentials are valid
3. Ensure `BANKING_API_BASE_URL` and `BANKING_API_TOKEN_URL` are correct

### API Calls Fail with 401

**Error**: "401 Unauthorized"

**Solutions**:
1. Token may have expired - the system should auto-refresh
2. Check token expiry in logs
3. Verify credentials have required scopes

### Registration Fails

**Error**: "Failed to register endpoint"

**Solutions**:
1. Check Postman collection file path
2. Verify JSON is valid
3. Check logs for specific parsing errors

## Postman Collection Format

The parser supports standard Postman collection v2.0 format:
- Folders for organization
- Requests with methods, URLs, headers, bodies
- Path parameters (`{param}` or `:param`)
- Query parameters
- Request body schemas (JSON, form-encoded, form-data)

## Security Best Practices

1. **Never commit credentials**: Keep `.env` in `.gitignore`
2. **Use environment variables**: Don't hardcode credentials
3. **Rotate credentials**: Regularly update client secrets
4. **Monitor token usage**: Check logs for authentication issues

## Files

- `src/core/utils/postman_parser.py` - Generic Postman collection parser
- `src/core/utils/oauth2_auth.py` - OAuth2 authentication manager
- `src/core/registry/tool_registry.py` - Tool registry with OAuth2 support
- `scripts/register_banking_apis.py` - Bulk registration script

