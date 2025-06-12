from fastapi import APIRouter, Depends
from app.schemas.schemas import HealthCheckResponse
from app.dependencies.dependencies import RAGServiceDep
from app.core.config import settings

router = APIRouter()

@router.get("/health", 
           response_model=HealthCheckResponse,
           summary="Health Check",
           description="Check the health status of all RAG system components.")
async def health_check(rag_service: RAGServiceDep):
    """Perform comprehensive health check."""
    services_status = rag_service.health_check()
    
    # Determine overall status
    overall_status = "healthy"
    if any(status in ["unhealthy", "not_initialized"] for status in services_status.values()):
        overall_status = "degraded"
    if all(status in ["unhealthy", "not_initialized", "not_configured"] for status in services_status.values()):
        overall_status = "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        services=services_status,
        version="0.4.0"
    )

@router.get("/ready",
           summary="Readiness Check", 
           description="Check if the service is ready to handle requests.")
async def readiness_check(rag_service: RAGServiceDep):
    """Check if service is ready."""
    services_status = rag_service.health_check()
    
    # Service is ready if at least one LLM is available
    is_ready = (services_status.get("azure_chat_llm") == "healthy" or 
                services_status.get("openrouter_llm") == "healthy") and \
               services_status.get("embeddings") == "healthy" and \
               services_status.get("vector_store") == "healthy"
    
    return {"ready": is_ready, "services": services_status}
