"""
Example: Bulk Register 100 Banking APIs
This script demonstrates how to register multiple banking APIs at once.
"""
import requests
import json
from typing import List, Dict, Any


def create_sample_apis() -> List[Dict[str, Any]]:
    """
    Create sample API configurations.
    Replace these with your actual 100 banking APIs.
    """
    return [
        # Account Management APIs
        {
            "api_name": "accounts.get_balance",
            "domain": "accounts",
            "description": "Get account balance for a specific account",
            "category": "account_management",
            "api_spec": {
                "method": "GET",
                "url": "https://api.bank.com/v1/accounts/{account_id}/balance",
                "headers": {"Content-Type": "application/json"},
                "request_schema": {
                    "account_id": {
                        "type": "string",
                        "required": True,
                        "description": "Account ID"
                    }
                },
                "path_params": ["account_id"]
            },
            "auth_config": {
                "type": "bearer",
                "token_source": "env_var",
                "env_var_name": "BANKING_API_TOKEN"
            },
            "execution_metadata": {
                "timeout": 30,
                "is_idempotent": True,
                "max_retries": 3
            },
            "risk_level": "read_only"
        },
        {
            "api_name": "accounts.get_statement",
            "domain": "accounts",
            "description": "Get account statement for a date range",
            "category": "account_management",
            "api_spec": {
                "method": "GET",
                "url": "https://api.bank.com/v1/accounts/{account_id}/statement",
                "request_schema": {
                    "account_id": {"type": "string", "required": True},
                    "start_date": {"type": "string", "required": True},
                    "end_date": {"type": "string", "required": True}
                },
                "path_params": ["account_id"],
                "query_params": ["start_date", "end_date"]
            },
            "auth_config": {"type": "bearer", "token_source": "env_var", "env_var_name": "BANKING_API_TOKEN"},
            "risk_level": "read_only"
        },
        # Payment APIs
        {
            "api_name": "payments.send_payment",
            "domain": "payments",
            "description": "Send a payment to another account",
            "category": "payment_processing",
            "api_spec": {
                "method": "POST",
                "url": "https://api.bank.com/v1/payments",
                "request_schema": {
                    "from_account": {"type": "string", "required": True},
                    "to_account": {"type": "string", "required": True},
                    "amount": {"type": "number", "required": True},
                    "currency": {"type": "string", "default": "USD"}
                }
            },
            "execution_metadata": {
                "timeout": 60,
                "is_idempotent": False,
                "requires_confirmation": True,
                "is_transactional": True
            },
            "risk_level": "financial"
        },
        {
            "api_name": "payments.get_payment_status",
            "domain": "payments",
            "description": "Get status of a payment transaction",
            "category": "payment_processing",
            "api_spec": {
                "method": "GET",
                "url": "https://api.bank.com/v1/payments/{payment_id}",
                "request_schema": {
                    "payment_id": {"type": "string", "required": True}
                },
                "path_params": ["payment_id"]
            },
            "risk_level": "read_only"
        },
        # Card APIs
        {
            "api_name": "cards.get_card_details",
            "domain": "cards",
            "description": "Get details of a credit/debit card",
            "category": "card_management",
            "api_spec": {
                "method": "GET",
                "url": "https://api.bank.com/v1/cards/{card_id}",
                "request_schema": {
                    "card_id": {"type": "string", "required": True}
                },
                "path_params": ["card_id"]
            },
            "risk_level": "read_only"
        },
        {
            "api_name": "cards.block_card",
            "domain": "cards",
            "description": "Block a card temporarily or permanently",
            "category": "card_management",
            "api_spec": {
                "method": "POST",
                "url": "https://api.bank.com/v1/cards/{card_id}/block",
                "request_schema": {
                    "card_id": {"type": "string", "required": True},
                    "reason": {"type": "string", "required": True},
                    "permanent": {"type": "boolean", "default": False}
                },
                "path_params": ["card_id"]
            },
            "execution_metadata": {
                "requires_confirmation": True
            },
            "risk_level": "write"
        }
        # Add 94 more APIs here...
    ]


def register_apis_bulk(api_base_url: str, apis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Register multiple APIs at once.
    
    Args:
        api_base_url: Base URL of the API (e.g., "http://localhost:8000")
        apis: List of API configurations
        
    Returns:
        Registration results
    """
    url = f"{api_base_url}/api/v1/tools/bulk-register"
    
    payload = {"tools": apis}
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        results = response.json()
        print(f"✅ Successfully registered {len(results)} tools")
        
        return {
            "success": True,
            "registered_count": len(results),
            "results": results
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error registering APIs: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def verify_registration(api_base_url: str, domain: str = None) -> None:
    """Verify registered tools by listing them."""
    url = f"{api_base_url}/api/v1/tools"
    params = {"domain": domain} if domain else {}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        tools = response.json()
        print(f"\n📋 Registered Tools ({len(tools)} total):")
        for tool in tools[:10]:  # Show first 10
            print(f"  - {tool['name']} ({tool['domain']})")
        if len(tools) > 10:
            print(f"  ... and {len(tools) - 10} more")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error listing tools: {e}")


def main():
    """Main function to register APIs."""
    # Configuration
    API_BASE_URL = "http://localhost:8000"  # Change to your server URL
    
    print("🚀 Starting bulk API registration...")
    print("=" * 60)
    
    # Create API configurations
    # Option 1: Use sample APIs (for testing)
    apis = create_sample_apis()
    
    # Option 2: Load from JSON file
    # with open('banking_apis.json', 'r') as f:
    #     apis = json.load(f)['tools']
    
    print(f"📝 Prepared {len(apis)} API configurations")
    
    # Register all APIs
    result = register_apis_bulk(API_BASE_URL, apis)
    
    if result["success"]:
        # Verify registration
        verify_registration(API_BASE_URL)
        
        print("\n" + "=" * 60)
        print("✅ Registration complete!")
        print(f"   Registered: {result['registered_count']} tools")
        print("\n💡 Next steps:")
        print("   1. Test a query: POST /api/v1/query")
        print("   2. List tools: GET /api/v1/tools")
        print("   3. Get tool details: GET /api/v1/tools/{tool_name}")
    else:
        print("\n❌ Registration failed. Check errors above.")


if __name__ == "__main__":
    main()

