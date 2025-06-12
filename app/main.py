import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes.routes import router as api_router_aggregator
from app.core.config import settings
from app.services.services import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting lifespan event: application startup...")

    app.state.rag_service = None

    try:
        print(f"ChromaDB Dir configuration from settings: {settings.CHROMA_DB_DIR}")

        chroma_db_file_path = os.path.join(settings.CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.isdir(settings.CHROMA_DB_DIR) or not os.path.exists(chroma_db_file_path):
            print(
                f"CRITICAL ERROR (startup): ChromaDB directory or database file ('chroma.sqlite3') not found at: {settings.CHROMA_DB_DIR}")
            print("Make sure the 'scripts/ingest_data.py' script has been run and successfully created the database.")
            raise RuntimeError(
                f"ChromaDB not found at {settings.CHROMA_DB_DIR}. Run ingest_data.py first."
            )

        app.state.rag_service = RAGService()
        print("RAGService successfully initialized during application startup and stored in app.state.rag_service.")
    except Exception as e:
        print(f"CRITICAL ERROR (startup): Failed to initialize RAGService: {e}")
        app.state.rag_service = None
        raise RuntimeError(f"Failed to initialize RAGService during startup: {e}") from e

    yield  # Application runs here

    print("Starting lifespan event: application shutdown...")
    if hasattr(app.state, 'rag_service'):
        app.state.rag_service = None  # Remove reference
    print("RAGService cleaned up (reference removed from app.state).")

app = FastAPI(
    title="RAG Application with FastAPI (Multi-LLM, Refactored)",
    description="API for RAG using Azure Embeddings and LLM from Azure OpenAI Chat or OpenRouter.",
    version="0.4.0",
    lifespan=lifespan
)

app.include_router(api_router_aggregator, prefix="/api/rag", tags=["RAG Queries"])

@app.get("/", summary="Root Endpoint", description="Welcome endpoint for RAG API.")
async def read_root():
    return {"message": "Welcome to the RAG API. Use endpoints under /api/rag/"}

