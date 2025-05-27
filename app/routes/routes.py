# File: app/routes/routes.py

from fastapi import APIRouter
from app.routes.open_ai.route import router as openai_rag_router
from app.routes.open_router.route import router as openrouter_rag_router

router = APIRouter()

router.include_router(
    openai_rag_router,
    prefix="/openai", # Endpoint akan menjadi /api/rag/openai/query
    tags=["RAG - Azure OpenAI Chat LLM"]
)
router.include_router(
    openrouter_rag_router,
    prefix="/openrouter", # Endpoint akan menjadi /api/rag/openrouter/query
    tags=["RAG - OpenRouter LLM"]
)

@router.get("/test", tags=["RAG System Test"])
async def rag_health_check():
    return {"status": "RAG route aggregator aktif"}