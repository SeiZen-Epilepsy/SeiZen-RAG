# File: app/main.py

import os
from fastapi import FastAPI, Request, HTTPException, status # Tambahkan Request, HTTPException, status
from contextlib import asynccontextmanager
from app.routes.routes import router as api_router_aggregator # Mengimpor router agregator
from app.core.config import settings
from app.services.services import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kode yang dijalankan sebelum aplikasi mulai menerima request (startup)
    print("Memulai event lifespan: startup aplikasi...")

    app.state.rag_service = None  # Inisialisasi default di app.state

    try:
        print(f"Konfigurasi ChromaDB Dir dari settings: {settings.CHROMA_DB_DIR}")

        chroma_db_file_path = os.path.join(settings.CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.isdir(settings.CHROMA_DB_DIR) or not os.path.exists(chroma_db_file_path):
            print(
                f"ERROR KRITIS (startup): Direktori atau file database ChromaDB ('chroma.sqlite3') tidak ditemukan di: {settings.CHROMA_DB_DIR}")
            print("Pastikan skrip 'scripts/ingest_data.py' sudah dijalankan dan berhasil membuat database.")
            raise RuntimeError(
                f"ChromaDB tidak ditemukan di {settings.CHROMA_DB_DIR}. Jalankan ingest_data.py terlebih dahulu."
            )

        # Inisialisasi RAGService dan simpan di app.state
        app.state.rag_service = RAGService()
        print("RAGService berhasil diinisialisasi selama startup aplikasi dan disimpan di app.state.rag_service.")
    except Exception as e:
        print(f"ERROR KRITIS (startup): Gagal menginisialisasi RAGService: {e}")
        app.state.rag_service = None  # Pastikan tetap None jika gagal
        raise RuntimeError(f"Gagal menginisialisasi RAGService selama startup: {e}") from e

    yield  # Aplikasi berjalan di sini

    # Kode yang dijalankan setelah aplikasi selesai menerima request (shutdown)
    print("Memulai event lifespan: shutdown aplikasi...")
    if hasattr(app.state, 'rag_service'):
        app.state.rag_service = None  # Hapus referensi
    print("RAGService di-cleanup (referensi dihapus dari app.state).")

app = FastAPI(
    title="RAG Application with FastAPI (Multi-LLM, Refactored)",
    description="API untuk RAG menggunakan Azure Embeddings dan LLM dari Azure OpenAI Chat atau OpenRouter.",
    version="0.4.0",
    lifespan=lifespan
)

app.include_router(api_router_aggregator, prefix="/api/rag", tags=["RAG Queries"])

@app.get("/", summary="Root Endpoint", description="Endpoint selamat datang untuk RAG API.")
async def read_root():
    return {"message": "Welcome to the RAG API. Gunakan endpoint di bawah /api/rag/"}

