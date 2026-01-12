from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from src.core.endpoints.health_endpoints import health_router
from src.core.endpoints.query_endpoints import query_router
from src.core.endpoints.tool_registry_endpoints import tool_registry_router
from settings import logger
from src.middleware.edge_security_middleware import prompt_security_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for FastAPI"""
    # Startup
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown - Close all connections
    logger.info("Application shutdown initiated - closing connections")
    try:
        # Close OpenSearch connections
        from src.core.utils.opensearch_connector import OpenSearchVectorSearchTool
        opensearch_tool = OpenSearchVectorSearchTool()
        opensearch_tool.close_connections()
        logger.info("OpenSearch connections closed")
        
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")
    
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Banking Agentic RAG Query Processing Service",
    description="Banking chatbot with 500+ API support - processes queries via REST API",
    version="1.0.0",
    openapi_prefix="/api/v1/rag-app",
    lifespan=lifespan
)

# Add middleware to the app
# Prompt security middleware (checks for prompt injection)
app.middleware("http")(prompt_security_middleware)

if __name__ == "__main__":
    logger.info("Starting Banking Agentic RAG Application")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Include routers
app.include_router(health_router)
app.include_router(query_router, prefix="/api/v1")
app.include_router(tool_registry_router, prefix="/api/v1")