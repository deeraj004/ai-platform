"""
RAG Tool - LangChain tool for Retrieval Augmented Generation.
"""
from typing import Dict, Any, List, Optional
from src.core.utils.query_processor import get_chunk_embeddings
from src.core.utils.opensearch_connector import vector_search_tool
from settings import logger, AZURE_OPENAI_SYSTEM_PROMPT, AZURE_OPENAI_USER_LOCATION_CONTEXT_TEMPLATE
from src.core.utils.azure_chatopenai import refine_and_augment_tool, generate_response_tool
from src.core.utils.reranker import cross_encoder_rerank_tool

# Error handler - create fallback if not available
try:
    from src.core.orchestrator.agent_event_logger import handle_agent_error, log_agent_start
except ImportError:
    # Fallback error handlers
    def handle_agent_error(state=None, error=None, agent_name="", additional_context=None, **kwargs):
        """Fallback error handler."""
        logger.error(f"{agent_name} error: {error}", exc_info=True)
        if state:
            state.exit_early = True
            if hasattr(state, 'metadata'):
                state.metadata["error"] = str(error)
        return {"error": str(error), "status": "error"}
    
    def log_agent_start(agent_name, *args, **kwargs):
        """Fallback logging function."""
        logger.info(f"{agent_name}: Starting")


class RagTool:
    """
    RAG Tool for Retrieval Augmented Generation.
    Performs document retrieval, query augmentation, reranking, and response generation.
    """
    
    def __init__(self):
        """Initialize RAG tool with required components."""
        self.vector_search_node = vector_search_tool
        self.query_refiner_node = refine_and_augment_tool
        self.reranker_node = cross_encoder_rerank_tool
        self.response_generator_node = generate_response_tool

    def retrieve_documents(self, user_query: str, alias_name: str) -> Dict[str, Any]:
        """Generate embeddings and retrieve documents from OpenSearch."""
        if not user_query or not alias_name:
            logger.warning("Missing 'query' or 'alias' in state.")
            return {
                "error": "Missing required query or alias information",
                "vector_search_results": {},
                "document_metadata": []
            }

        # Generate embeddings
        try:
            embedding_query = get_chunk_embeddings.invoke(user_query)
            logger.info(f"Generated embeddings for user query: {user_query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {
                "error": f"Embedding generation failed: {str(e)}",
                "vector_search_results": {},
                "document_metadata": []
            }

        # Retrieve documents from OpenSearch
        try:
            vector_search_results = vector_search_tool.invoke({
                "alias_name": alias_name,
                "query_vector": embedding_query,
                "query_filter_attributes": {},
            })
            chunk_data_count = len(vector_search_results.get('chunk_data', []))
            logger.info(f"Retrieved {chunk_data_count} chunks from OpenSearch.")
            
            # Extract document metadata for citation building
            document_metadata = vector_search_results.get('document_metadata', [])
            if document_metadata:
                logger.info(f"Stored {len(document_metadata)} document metadata entries for citations")
            
            if chunk_data_count == 0:
                logger.warning("No documents retrieved from OpenSearch")
                
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return {
                "error": f"Document retrieval failed: {str(e)}",
                "vector_search_results": {},
                "document_metadata": []
            }
            
        return {
            "vector_search_results": vector_search_results,
            "document_metadata": document_metadata
        }

    def augment_query(
        self, 
        user_query: str, 
        conversation_history: List[Dict[str, Any]], 
        precheck_summary: Dict[str, Any]
    ) -> List[str]:
        """
        Augment user query with conversation history context.
        """
        try:
            augmented_query_result = refine_and_augment_tool.invoke({
                "query": user_query,
                "precheck_summary": precheck_summary,
                "conversation_history": conversation_history or [],
                "template_kwargs": {}
            })
            query_lines = augmented_query_result.get("refined_query", "").split("\n") if augmented_query_result.get("refined_query") else [user_query]
            if not query_lines:
                logger.warning("Query augmentation returned empty result, using original query")
                augmented_user_query = [user_query]
            else:
                augmented_user_query = query_lines
            logger.info(f"Query augmented successfully: {augmented_user_query[0][:50]}...")

        except Exception as e:
            logger.warning(f"Query augmentation failed: {e}, using original query")
            augmented_user_query = [user_query]
            
        return augmented_user_query
        
    def generate_response(
        self, 
        user_query: str, 
        augmented_user_query: List[str], 
        vector_search_results: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None,
        user_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response based on the user's query and the retrieved documents.
        """
        try:
            document_metadata = vector_search_results.get('document_metadata', [])
            
            # Rerank documents
            reranked_results = cross_encoder_rerank_tool.invoke({
                "query": user_query,
                "source_urls": vector_search_results.get('source_url', []),
                "chunk_data": vector_search_results.get('chunk_data', []),
                "scores": vector_search_results.get('scores', []),
                "query_filter_attributes": {},
                "document_metadata": document_metadata
            })
            logger.info("Document reranking completed")
            
            # Update document_metadata with reranked order
            if isinstance(reranked_results, tuple) and len(reranked_results) >= 4:
                document_metadata = reranked_results[3]
                logger.info(f"Updated document_metadata with {len(document_metadata)} reranked entries")
            
            # Extract source_urls and chunk_data from reranked_results
            if isinstance(reranked_results, tuple) and len(reranked_results) >= 4:
                source_urls, chunk_data, _, _ = reranked_results
            elif isinstance(reranked_results, tuple) and len(reranked_results) == 3:
                source_urls, chunk_data, _ = reranked_results
            else:
                source_urls = vector_search_results.get('source_url', [])
                chunk_data = vector_search_results.get('chunk_data', [])
                logger.warning(f"Unexpected reranked_results format: {type(reranked_results)}, using vector_search_results")

            # Determine system prompt
            system_prompt = AZURE_OPENAI_SYSTEM_PROMPT
            
            # Check for custom system prompt from additional_params
            if additional_params:
                custom_prompt = additional_params.get("custom_system_prompt")
                if custom_prompt:
                    system_prompt = custom_prompt
                    logger.info("Using custom system prompt from additional_params")
                
                # Check if user location filtering is required
                requires_user_location_filtering = additional_params.get("requires_user_location_filtering", False)
                if requires_user_location_filtering and user_info:
                    user_country_code = user_info.get("country_code", "")
                    user_country_name = user_info.get("country", "")
                    system_prompt = system_prompt + AZURE_OPENAI_USER_LOCATION_CONTEXT_TEMPLATE.format(
                        user_country_name=user_country_name, 
                        user_country_code=user_country_code
                    )
                    logger.info(f"[LOCATION_FILTERING] Injected user location: {user_country_name} ({user_country_code})")

            response_result = generate_response_tool.invoke({
                "query": user_query,
                "context": augmented_user_query[0] if augmented_user_query else user_query,
                "ranked_search_results": (source_urls, chunk_data),
                "template_kwargs": {},
                "custom_system_prompt": system_prompt,
            })
            
            # Extract answer from response
            answer = response_result.get("final_response", "") if isinstance(response_result, dict) else str(response_result)
            logger.info(f"Response generation completed. Answer length: {len(answer) if answer else 0}")
            logger.debug(f"Answer content: {answer[:100] if answer else 'None'}...")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return ""
        
        return answer      

    def run(
        self, 
        user_query: str, 
        alias_name: str, 
        conversation_history: List[Dict[str, Any]], 
        precheck_summary: Dict[str, Any], 
        additional_params: Optional[Dict[str, Any]] = None, 
        user_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline with the given parameters.
        """
        log_agent_start("RagTool", user_query, alias_name, None, conversation_history, precheck_summary, additional_params, user_info)
        
        try:
            # Execute RAG pipeline steps
            retrieval_result = self.retrieve_documents(user_query, alias_name)
            
            if "error" in retrieval_result:
                return retrieval_result
                
            vector_search_results = retrieval_result["vector_search_results"]
            augmented_user_query = self.augment_query(user_query, conversation_history, precheck_summary)
            
            answer = self.generate_response(
                user_query, 
                augmented_user_query, 
                vector_search_results,
                additional_params=additional_params,
                user_info=user_info
            )
            
            return {
                "answer": answer,
                "documents_retrieved": len(vector_search_results.get('chunk_data', [])) if vector_search_results else 0,
                "has_answer": bool(answer),
                "status": "success"
            }
            
        except Exception as e:
            return handle_agent_error(
                user_query=user_query,
                alias_name=alias_name,
                conversation_history=conversation_history,
                precheck_summary=precheck_summary,
                error=e,
                agent_name="RagTool"
            )
