"""
Basic Authentication for FastAPI
Protects Swagger UI and API endpoints with HTTP Basic Auth.
"""
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
import secrets
from settings import (
    SWAGGER_USERNAME,
    SWAGGER_PASSWORD,
    logger
)

# Create HTTPBasic security scheme
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Security(security)) -> str:
    """
    Verify HTTP Basic Auth credentials.
    
    Args:
        credentials: HTTP Basic Auth credentials from request
    
    Returns:
        Username if credentials are valid
    
    Raises:
        HTTPException: If credentials are invalid
    """
    # Get credentials from environment or use defaults
    correct_username = secrets.compare_digest(credentials.username, SWAGGER_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, SWAGGER_PASSWORD)
    
    if not (correct_username and correct_password):
        logger.warning(f"Failed authentication attempt from user: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    logger.debug(f"Successful authentication for user: {credentials.username}")
    return credentials.username


def get_current_user(credentials: HTTPBasicCredentials = Security(security)) -> str:
    """
    Dependency to get current authenticated user.
    Use this in route dependencies to protect endpoints.
    
    Example:
        @app.get("/protected")
        def protected_route(user: str = Depends(get_current_user)):
            return {"message": f"Hello {user}"}
    """
    return verify_credentials(credentials)

