from fastapi import Request, HTTPException, status, Depends
from typing import Annotated
from app.services.services import RAGService
from app.controllers.open_ai.controller import OpenAIRAGController
from app.controllers.open_router.controller import OpenRouterRAGController

def get_rag_service_from_state(request: Request) -> RAGService:
    """Get RAGService from application state."""
    if not hasattr(request.app.state, 'rag_service') or request.app.state.rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not available (not initialized by lifespan)."
        )
    return request.app.state.rag_service

RAGServiceDep = Annotated[RAGService, Depends(get_rag_service_from_state)]

def get_openai_rag_controller(service: RAGServiceDep) -> OpenAIRAGController:
    """Get OpenAI RAG controller instance."""
    return OpenAIRAGController(rag_service=service)

def get_openrouter_rag_controller(service: RAGServiceDep) -> OpenRouterRAGController:
    """Get OpenRouter RAG controller instance."""
    return OpenRouterRAGController(rag_service=service)

OpenAIControllerDep = Annotated[OpenAIRAGController, Depends(get_openai_rag_controller)]
OpenRouterControllerDep = Annotated[OpenRouterRAGController, Depends(get_openrouter_rag_controller)]
