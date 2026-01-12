"""
Query Service
Simplified query service for banking chatbot - processes queries directly via API.
"""
from typing import Dict, Any

from src.core.orchestrator.agent_orchestrator import agent_orchestrator
from settings import logger, OPENSEARCH_SEARCH_ALIAS_NAME


class QueryService:
    """
    Simplified Query Service for Banking Chatbot.
    
    This service processes queries directly without DynamoDB, SQS, or SNS dependencies.
    Queries are received via API endpoints and processed synchronously.
    """
    
    def __init__(self):
        logger.info("Initializing QueryService...")
        try:
            self.orchestrator = agent_orchestrator
            logger.info("Agent orchestrator initialized")
            self.logger = logger
            logger.info("QueryService initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QueryService: {e}")
            raise
    
    async def process_query(
        self,
        request_id: str,
        conversation_id: str,
        session_id: str,
        user_query: str,
        conversation_history: list = None,
        user_id: str = "",
        open_search_alias_name: str = None
    ) -> Dict[str, Any]:
        """
        Process a query through the agentic workflow.
        
        Args:
            request_id: Unique request identifier
            conversation_id: Conversation session identifier
            session_id: User session identifier
            user_query: User's query message
            conversation_history: Previous conversation messages (optional)
            user_id: User identifier (optional)
            open_search_alias_name: OpenSearch alias (optional)
            
        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(f"Processing query: request_id={request_id}, conversation_id={conversation_id}")
            
            # Determine OpenSearch alias
            opensearch_alias = open_search_alias_name or OPENSEARCH_SEARCH_ALIAS_NAME
            
            # Build query payload
            query_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "conversation_session_id": conversation_id,
                "request_id": request_id,
                "query": user_query,
                "conversation_history": conversation_history or [],
                "open_search_alias_name": opensearch_alias,
                "open_search_index_name": opensearch_alias,
                "additional_params": {}
            }
            
            # Process through orchestrator
            response_text = await self.orchestrator.orchestrate_query(query_payload)
            
            return {
                "request_id": request_id,
                "conversation_id": conversation_id,
                "response": response_text,
                "status": "COMPLETED"
            }
            
        except Exception as e:
            logger.error(f"Error processing query request_id={request_id}: {e}", exc_info=True)
            return {
                "request_id": request_id,
                "conversation_id": conversation_id,
                "response": f"An error occurred while processing your request: {str(e)}",
                "status": "FAILED",
                "error": str(e)
            }


# Global query service instance
query_service = QueryService()
