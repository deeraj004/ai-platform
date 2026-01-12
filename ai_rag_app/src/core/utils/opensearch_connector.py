# src/core/tools/base/retrievers/opensearch_connector.py
import json
import os
import boto3
from typing import Dict, Any
from langchain_core.tools import tool
from requests_aws4auth import AWS4Auth
from opensearchpy import (
    OpenSearch,
    RequestsHttpConnection,
    OpenSearchException,
    ConnectionError,
    TransportError,
)
from settings import (
    OPENSEARCH_SEARCH_FIELDS_BY_ALIAS,
    OPENSEARCH_SEARCH_RECORDS_BY_ALIAS,
    OPENSEARCH_TOP_K,
    logger,
    AWS_REGION,
    AWS_SERVICE_NAME,
    IS_CLOUD_ENABLED,
    OPENSEARCH_HOST,
)
from src.core.orchestrator.state_schema import AgenticState

class OpenSearchVectorSearchTool:
    """
    Singleton wrapper for OpenSearch vector search.
    Implements singleton pattern and connection pooling.
    """

    _instance = None
    _lock = None  # Will be initialized as threading.Lock()
    _client_cache = {}  # Cache OpenSearch clients

    def __new__(cls):
        """Implement thread-safe singleton pattern"""
        import threading
        if cls._lock is None:
            cls._lock = threading.Lock()
            
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OpenSearchVectorSearchTool, cls).__new__(cls)
                logger.info("Created new OpenSearchVectorSearchTool singleton instance")
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.region_prefix = os.getenv("REGION_PREFIX", "BANKING-")
        self.aws_region = AWS_REGION
        self.aws_service_name = AWS_SERVICE_NAME
        self.is_cloud_enabled = IS_CLOUD_ENABLED
        self._initialized = True
        logger.info("OpenSearchVectorSearchTool initialized")

    def get_opensearch_auth(self):
        """Get AWS4Auth credentials for OpenSearch."""
        try:
            session = boto3.Session(region_name=self.aws_region)
            credentials = session.get_credentials()
            if not credentials or not credentials.access_key:
                raise Exception("IAM role credentials not found.")
        except Exception:
            session = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=self.aws_region)
            credentials = session.get_credentials()
            if not credentials or not credentials.access_key:
                raise RuntimeError("No valid AWS credentials found.")

        region = self.aws_region or session.region_name
        if not region:
            raise RuntimeError("AWS region not found.")

        logger.info(f" Retrieved AWS credentials from boto3 session")
        logger.debug(f"AWS Access Key: {credentials.access_key[:4]}********")
        logger.debug(f"Region: {self.aws_region}")

        return AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            self.aws_service_name,
            session_token=credentials.token,
        )

    def get_opensearch_client(self):
        """Return OpenSearch client with AWS SigV4 authentication (cached)."""
        host = (
            OPENSEARCH_HOST
            if self.is_cloud_enabled
            else os.getenv("OPENSEARCH_HOST")
        )
        
        # Check if client is already cached
        client_key = f"{host}:{self.aws_region}"
        if client_key in self._client_cache:
            logger.debug(f"Reusing cached OpenSearch client for {host}")
            return self._client_cache[client_key]
        
        # Check if this is a local OpenSearch instance
        is_local = host in ["localhost", "127.0.0.1"]
        
        if is_local:
            # Local OpenSearch without authentication
            logger.info(f"Creating new LOCAL OpenSearch client for: {host}")
            client = OpenSearch(
                hosts=[{"host": host, "port": 9200}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                connection_class=RequestsHttpConnection,
                pool_maxsize=10,
                timeout=30,
            )
        else:
            # Cloud OpenSearch with AWS authentication
            auth = self.get_opensearch_auth()
            logger.info(f"Creating new CLOUD OpenSearch client for: {host} in {self.aws_region}")
            client = OpenSearch(
                hosts=[{"host": host, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                http_compress=True,
                connection_class=RequestsHttpConnection,
                pool_maxsize=10,  # Connection pool size
                timeout=30,  # Request timeout
            )
        
        # Cache the client
        self._client_cache[client_key] = client
        return client
    
    def close_connections(self):
        """Close all OpenSearch client connections"""
        try:
            for client_key, client in self._client_cache.items():
                try:
                    if hasattr(client, 'close'):
                        client.close()
                    logger.debug(f"Closed OpenSearch client: {client_key}")
                except Exception as e:
                    logger.error(f"Error closing OpenSearch client {client_key}: {e}")
            self._client_cache.clear()
            logger.info("All OpenSearch connections closed")
        except Exception as e:
            logger.error(f"Error closing OpenSearch connections: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connections()
        return False

    def build_region_filters(self, query_filter_attributes: dict = None) -> dict:
        """Builds region/country filters for search queries."""
        should_terms = []
        if query_filter_attributes and query_filter_attributes.get("country_code"):
            region_fields = [
                ("region_name", "region"),
                ("region_code", "region_code"),
                ("country_name", "country"),
                ("country_code", "country_code"),
            ]
            for user_key, _ in region_fields:
                value = query_filter_attributes.get(user_key, "")
                if isinstance(value, str) and value:
                    formatted = f"{self.region_prefix}{value.replace(' ', '').upper()}"
                    should_terms.append({"term": {"country": formatted}})
                elif isinstance(value, list):
                    for item in value:
                        if item and isinstance(item, str):
                            formatted = f"{self.region_prefix}{item.replace(' ', '').upper()}"
                            should_terms.append({"term": {"country": formatted}})

        should_terms.append({"term": {"country": self.global_region_name}})
        return {"bool": {"should": should_terms, "minimum_should_match": 1}}

    def _process_search_response(self, response: dict, alias_name: str) -> dict:
        """Parse OpenSearch response into structured state with document metadata for citations."""
        if "hits" in response:
            chunk_data_list = [hit["_source"]["chunk_data"] for hit in response["hits"]["hits"]]
            source_url_list = list({
                hit.get("_source", {}).get("source_url")
                for hit in response["hits"]["hits"]
            })
            score_list = [float(hit.get("_score", 0)) for hit in response["hits"]["hits"]]
            
            # Extract document metadata for citations
            document_metadata_list = []
            for hit in response["hits"]["hits"]:
                source = hit.get("_source", {})
                metadata = source.get("metadata", {})
                
                # Build metadata dict with all citation-relevant fields
                doc_meta = {
                    "doc_id": hit.get("_id"),  # OpenSearch document ID
                    "source_url": source.get("source_url"),
                    "file_name": metadata.get("source_file") or source.get("document_name") or "Unknown Document",
                    "page_number": metadata.get("page_number") or source.get("page_number"),
                    "chunk_index": source.get("chunk_index"),
                    "score": float(hit.get("_score", 0))
                }
                document_metadata_list.append(doc_meta)
            
            logger.info(f"Search by alias '{alias_name}' completed. Found {len(chunk_data_list)} results with metadata.")
            return {
                "chunk_data": chunk_data_list, 
                "source_url": source_url_list, 
                "scores": score_list,
                "document_metadata": document_metadata_list
            }
        else:
            logger.error(f"Search failed. Response: {response}")
            return {"chunk_data": [], "source_url": [], "scores": [], "document_metadata": []}


    def _search_by_alias_impl(self, alias_name: str, query_vector: list, query_filter_attributes: dict = {}) -> Any:  
        """Internal implementation: takes state, searches OpenSearch, returns results."""
        # Extract specific data needed by the tool (following preprocessing agent pattern)
        
        logger.info(f"Starting search_by_alias for alias: {alias_name}")

        try:
            fields_str = OPENSEARCH_SEARCH_FIELDS_BY_ALIAS
            field_list = fields_str.split(",")
            search_query = json.loads(OPENSEARCH_SEARCH_RECORDS_BY_ALIAS)

            search_query["query"]["bool"]["must"][0]["knn"]["vector"]["vector"] = query_vector
            search_query["query"]["bool"]["must"][0]["knn"]["vector"]["k"] = int(OPENSEARCH_TOP_K)
            search_query["_source"] = field_list

            if query_filter_attributes:
                region_clause = self.build_region_filters(query_filter_attributes)
                search_query["query"]["bool"]["filter"] = region_clause
                logger.info(f"Adding region filter to search query: {region_clause}")
            else:
                search_query["query"]["bool"].pop("filter", None)

            client = self.get_opensearch_client()
            response = client.search(index=alias_name, body=search_query)
            return self._process_search_response(response, alias_name)

        except (ConnectionError, TransportError, OpenSearchException) as e:
            logger.error(f"OpenSearch client error: {e}")
            return {"chunk_data": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Error searching index by alias '{alias_name}': {e}")
            return {"chunk_data": [], "error": str(e)}


@tool(description="Searches documents in OpenSearch by alias using vector similarity.")
def search_by_alias(
    alias_name: str,
    query_vector: list,
    query_filter_attributes: dict = {},
) -> Any:
    return opensearch_vector_db._search_by_alias_impl(alias_name, query_vector, query_filter_attributes)


# Singleton instance
opensearch_vector_db = OpenSearchVectorSearchTool()

# Exposed tool
vector_search_tool = search_by_alias
