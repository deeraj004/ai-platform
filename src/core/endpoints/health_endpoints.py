"""
Health Check Endpoints
Provides comprehensive health monitoring
"""
import sys
import os
import time
import psutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

# Add project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.services.query_service import query_service
from settings import logger

# Create health router
health_router = APIRouter(tags=["Health"])

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    environment: str

class DetailedHealthResponse(BaseModel):
    """Detailed health check response model"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    environment: str
    system: Dict[str, Any]
    services: Dict[str, str]
    dependencies: Dict[str, str]

class ReadinessResponse(BaseModel):
    """Readiness check response model"""
    status: str
    ready: bool
    checks: Dict[str, bool]
    timestamp: str

# Application start time for uptime calculation
app_start_time = time.time()

def get_uptime() -> float:
    """Get application uptime in seconds"""
    return time.time() - app_start_time

def check_query_service() -> bool:
    """Check if query service is operational"""
    try:
        # Check if query service is initialized
        return query_service is not None and hasattr(query_service, 'orchestrator')
    except Exception as e:
        logger.error(f"Query service check failed: {e}")
        return False

def get_system_metrics() -> Dict[str, Any]:
    """Get basic system metrics"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    except Exception as e:
        logger.warning(f"Could not get system metrics: {e}")
        return {"error": "System metrics unavailable"}

@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint for load balancers and monitoring systems.
    Returns 200 OK if the service is healthy, 503 Service Unavailable otherwise.
    """
    try:
        # Basic health checks
        service_ok = check_query_service()
        
        if not service_ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service dependencies are not healthy"
            )
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",  # You can get this from settings or version file
            uptime_seconds=get_uptime(),
            environment=os.getenv("DEPLOYMENT_ENV", "development"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@health_router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with system metrics and service status.
    Useful for debugging and comprehensive monitoring.
    """
    try:
        # Comprehensive health checks
        service_ok = check_query_service()
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        # Check service dependencies
        dependencies = {
            "query_service": "healthy" if service_ok else "unhealthy",
            "aws_services": "healthy",  # You can add actual AWS health checks here
            "database": "healthy"      # You can add actual DB health checks here
        }
        
        # Determine overall status
        overall_status = "healthy" if all(
            status == "healthy" for status in dependencies.values()
        ) else "degraded"
    
        return DetailedHealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",
            uptime_seconds=get_uptime(),
            environment=os.getenv("ENVIRONMENT", "development"),
            system=system_metrics,
            services={
                "query_processing": "running",
                "api_endpoints": "active",
            },
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detailed health check failed: {str(e)}"
        )

@health_router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check for Kubernetes and container orchestration.
    Indicates if the service is ready to accept traffic.
    """
    try:
        checks = {
            "query_service_ready": check_query_service(),
            "dependencies_available": True  # Add more dependency checks as needed
        }
        
        ready = all(checks.values())
        
        return ReadinessResponse(
            status="ready" if ready else "not_ready",
            ready=ready,
            checks=checks,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return ReadinessResponse(
            status="not_ready",
            ready=False,
            checks={"error": str(e)},
            timestamp=datetime.now(timezone.utc).isoformat()
        )

@health_router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes and container orchestration.
    Simple endpoint that indicates the service is alive.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": get_uptime()
    }

@health_router.get("/status")
async def status_check():
    """
    General status endpoint with basic service information.
    """
    try:
        return {
            "service": "RAG Query Processing Service",
            "status": "operational",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "version": "1.0.0",
            "uptime_seconds": get_uptime(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoints": {
                "health": "/health",
                "detailed_health": "/health/detailed", 
                "readiness": "/ready",
                "liveness": "/live",
                "status": "/status"
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )
