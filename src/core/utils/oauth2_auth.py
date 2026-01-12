"""
OAuth2 Authentication Manager
Generic OAuth2 client credentials flow for banking APIs.
Handles access token generation and management.
"""
import httpx
import time
from typing import Optional, Dict, Any
from settings import logger, BANKING_API_CLIENT_ID, BANKING_API_CLIENT_SECRET, BANKING_API_BASE_URL, BANKING_API_TOKEN_URL


class OAuth2AuthManager:
    """
    Manages OAuth2 access tokens using client credentials flow.
    Thread-safe singleton for token management.
    Generic implementation for any banking API.
    """
    _instance = None
    _lock = None
    
    def __new__(cls):
        import threading
        if cls._lock is None:
            cls._lock = threading.Lock()
        
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OAuth2AuthManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        import threading
        with self._lock:
            if hasattr(self, '_initialized'):
                return
            
            self._access_token: Optional[str] = None
            self._token_expires_at: float = 0
            self._token_lock = threading.Lock()
            self._base_url = BANKING_API_BASE_URL or ""
            self._token_url = BANKING_API_TOKEN_URL or f"{self._base_url}/v1/oauth2/token"
            self._client_id = BANKING_API_CLIENT_ID
            self._client_secret = BANKING_API_CLIENT_SECRET
            self._initialized = True
    
    def get_access_token(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get valid OAuth2 access token, refreshing if needed.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Access token string or None if authentication failed
        """
        with self._token_lock:
            # Check if token is still valid
            if not force_refresh and self._access_token and time.time() < self._token_expires_at:
                return self._access_token
            
            # Refresh token
            return self._refresh_token()
    
    def _refresh_token(self) -> Optional[str]:
        """Refresh OAuth2 access token using client credentials flow."""
        if not self._client_id or not self._client_secret:
            logger.warning("Banking API credentials not configured. Set BANKING_API_CLIENT_ID and BANKING_API_CLIENT_SECRET in environment.")
            return None
        
        if not self._token_url:
            logger.warning("Token URL not configured. Set BANKING_API_TOKEN_URL in environment.")
            return None
        
        try:
            # Use Basic Auth with client_id:client_secret
            auth = (self._client_id, self._client_secret)
            data = {"grant_type": "client_credentials"}
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self._token_url,
                    auth=auth,
                    data=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self._access_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
                    # Set expiry 5 minutes before actual expiry for safety
                    self._token_expires_at = time.time() + expires_in - 300
                    
                    logger.info(f"OAuth2 access token refreshed. Expires in {expires_in}s")
                    return self._access_token
                else:
                    logger.error(f"Failed to refresh OAuth2 token: {response.status_code} - {response.text}")
                    return None
        
        except Exception as e:
            logger.error(f"Error refreshing OAuth2 token: {e}", exc_info=True)
            return None
    
    def invalidate_token(self):
        """Invalidate current token (force refresh on next request)."""
        with self._token_lock:
            self._access_token = None
            self._token_expires_at = 0


# Global instance
_oauth2_auth_manager: Optional[OAuth2AuthManager] = None


def get_oauth2_auth_manager() -> OAuth2AuthManager:
    """Get global OAuth2 auth manager instance."""
    global _oauth2_auth_manager
    if _oauth2_auth_manager is None:
        _oauth2_auth_manager = OAuth2AuthManager()
    return _oauth2_auth_manager


def get_oauth2_access_token() -> Optional[str]:
    """Get OAuth2 access token (convenience function)."""
    return get_oauth2_auth_manager().get_access_token()

