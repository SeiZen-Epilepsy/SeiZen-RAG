from fastapi import APIRouter, HTTPException, status
from app.schemas.schemas import QueryRequest, QueryResponse
from app.dependencies.dependencies import OpenAIControllerDep
from app.core.logging_config import logger

router = APIRouter()

@router.post("/query",
             response_model=QueryResponse,
             summary="Submit Question to RAG (Azure OpenAI Chat LLM)",
             description="Send a question to the RAG system, using Azure OpenAI Chat LLM for answer generation.")
async def ask_rag_openai_azure(
        request_data: QueryRequest,
        controller: OpenAIControllerDep
):
    if not request_data.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    try:
        logger.info(f"Receiving query for Azure OpenAI Chat LLM: {request_data.question}")
        result = await controller.handle_query(request_data.question)
        return QueryResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        logger.error(f"Unexpected error (OpenAI/Azure RAG Route): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")