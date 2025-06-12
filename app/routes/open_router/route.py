# File: app/routes/open_router/route.py

from fastapi import APIRouter, HTTPException, status
from app.schemas.schemas import QueryRequest, QueryResponse
from app.dependencies.dependencies import OpenRouterControllerDep
from app.core.logging_config import logger

router = APIRouter()

@router.post("/query",
             response_model=QueryResponse,
             summary="Submit Question to RAG (OpenRouter LLM)",
             description="Send a question to the RAG system, using OpenRouter LLM for answer generation.")
async def ask_rag_openrouter(
    request_data: QueryRequest,
    controller: OpenRouterControllerDep
):
    if not request_data.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    try:
        logger.info(f"Receiving query for OpenRouter LLM: {request_data.question}")
        result = await controller.handle_query(request_data.question)
        return QueryResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        logger.error(f"Unexpected error (OpenRouter RAG Route): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")

