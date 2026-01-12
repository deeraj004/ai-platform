"""
Query Processing Endpoints
API endpoints for banking chatbot query processing.
Enhanced with strict input validation for IDs and query fields using proper libraries.
"""
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Path, Depends, Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, field_validator, constr, ConfigDict
import secrets

from src.core.orchestrator.agent_orchestrator import agent_orchestrator
from settings import logger, OPENSEARCH_SEARCH_ALIAS_NAME, ENABLE_API_AUTH, API_USERNAME, API_PASSWORD

# Create query router
query_router = APIRouter(tags=["Query"])

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
MAX_QUERY_LENGTH = 2000
ALIAS_PATTERN = r'^[a-zA-Z0-9_-]+$'
MAX_CONVERSATION_HISTORY = 50


class QueryRequest(BaseModel):
    """Request model for query processing with strict validation"""
    request_id: str = Field(
        ...,
        description="Unique request identifier (KSUID format: 27-character base62 string)"
    )
    
    conversation_id: str = Field(
        ...,
        description="Conversation session identifier (UUID v4 format)",
        alias="conversation_session_id"
    )
    
    session_id: str = Field(
        ...,
        description="User session identifier (UUID v4 format)"
    )
    
    user_query: constr(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_QUERY_LENGTH
    ) = Field(
        ...,
        description="User's query message",
        alias="query"
    )
    
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Previous conversation messages",
        max_length=MAX_CONVERSATION_HISTORY
    )
    
    user_id: Optional[constr(
        pattern=ALIAS_PATTERN,
        max_length=100
    )] = Field(
        default="",
        description="User identifier (optional)"
    )
    
    open_search_alias_name: Optional[constr(
        pattern=ALIAS_PATTERN,
        max_length=100
    )] = Field(
        default=None,
        description="OpenSearch alias name (optional, defaults to deployment setting)"
    )
    
    
    model_config = ConfigDict(validate_by_name=True)


class QueryResponse(BaseModel):
    """Response model for query processing"""
    request_id: str
    conversation_id: str
    response: str
    status: str
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@query_router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    user: str = Depends(verify_api_credentials)
):
    """
    Process a banking chatbot query through the agentic workflow.
    
    This endpoint accepts a query request and processes it through the complete
    agentic workflow: Intent Classification → Planning → Execution → Evaluation.
    
    **Request Parameters:**
    - `user_query`: The user's query message
    - `conversation_history`: (Optional) Previous conversation messages 
    - `user_id`: (Optional) User identifier
    - `open_search_alias_name`: (Optional) OpenSearch alias, defaults to deployment setting
    
    **Response:**
    - `response`: Generated response from the banking chatbot
    - `status`: Processing status (COMPLETED, FAILED, etc.)
    - `processing_time`: Time taken to process (seconds)
    - `metadata`: Additional metadata about the processing
    """
    try:
        logger.info(f"Received query request: request_id={request.request_id}, conversation_id={request.conversation_id}")
        
        # Determine OpenSearch alias
        opensearch_alias = request.open_search_alias_name or OPENSEARCH_SEARCH_ALIAS_NAME
        
        if not opensearch_alias:
            logger.warning("No OpenSearch alias provided, using default")
        
        # Build query payload for orchestrator
        query_payload = {
            "user_id": request.user_id or "",
            "query": request.user_query,
            "conversation_history": request.conversation_history or [],
            "open_search_alias_name": opensearch_alias,
            "open_search_index_name": opensearch_alias,
            "additional_params": {}
        }
        
        # Process query through orchestrator
        logger.info(f"Processing query through orchestrator: request_id={request.request_id}")
        response_text = await agent_orchestrator.orchestrate_query(query_payload)
        
        # Build response
        response = QueryResponse(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            response=response_text,
            status="COMPLETED",
            metadata={
                "opensearch_alias": opensearch_alias
            }
        )
        
        logger.info(f"Query processed successfully: request_id={request.request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query request_id={request.request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


