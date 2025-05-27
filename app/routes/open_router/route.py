# File: app/routes/open_router/route.py

from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Annotated
from app.schemas.schemas import QueryRequest, QueryResponse
from app.controllers.open_router.controller import OpenRouterRAGController
from app.services.services import RAGService

router = APIRouter()

# Dependency untuk mendapatkan RAGService dari app.state (bisa di-share jika dipindah ke dependencies.py)
def get_rag_service_from_state(request: Request) -> RAGService:
    if not hasattr(request.app.state, 'rag_service') or request.app.state.rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Layanan RAG tidak tersedia (tidak diinisialisasi oleh lifespan)."
        )
    return request.app.state.rag_service

RAGServiceDep = Annotated[RAGService, Depends(get_rag_service_from_state)]

# Dependency untuk OpenRouterRAGController
def get_openrouter_rag_controller(service: RAGServiceDep) -> OpenRouterRAGController:
    return OpenRouterRAGController(rag_service=service)

OpenRouterControllerDep = Annotated[OpenRouterRAGController, Depends(get_openrouter_rag_controller)]

@router.post("/query",
             response_model=QueryResponse,
             summary="Ajukan Pertanyaan ke RAG (OpenRouter LLM)",
             description="Mengirimkan pertanyaan ke sistem RAG, menggunakan OpenRouter LLM untuk generasi jawaban.")
async def ask_rag_openrouter(
    request_data: QueryRequest,
    controller: OpenRouterControllerDep
):
    if not request_data.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pertanyaan tidak boleh kosong.")

    try:
        print(f"Menerima query untuk OpenRouter LLM: {request_data.question}")
        result = await controller.handle_query(request_data.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        print(f"Error tidak terduga (OpenRouter RAG Route): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Terjadi kesalahan internal.")

