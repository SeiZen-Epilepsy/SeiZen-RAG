# File: app/main.py

from fastapi import FastAPI, HTTPException, status, Request
from contextlib import asynccontextmanager
import os
from typing import Optional

try:
    # Menggunakan nama file yang diberikan pengguna
    from .api.routes import routes as api_routes
    from .core.config import settings
    from .services.services import RAGService
except ImportError as e:
    print(f"ImportError di main.py: {e}")
    # Fallback jika diperlukan, sesuaikan dengan cara Anda menjalankan
    from api.routes import routes as api_routes_fallback
    from core.config import settings as settings_fallback
    from services.services import RAGService as RAGService_fallback

    api_routes = api_routes_fallback
    settings = settings_fallback
    RAGService = RAGService_fallback


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
            raise RuntimeError(  # Melempar error akan menghentikan startup aplikasi
                f"ChromaDB tidak ditemukan di {settings.CHROMA_DB_DIR}. Jalankan ingest_data.py terlebih dahulu."
            )

        # Inisialisasi RAGService dan simpan di app.state
        app.state.rag_service = RAGService()
        print("RAGService berhasil diinisialisasi selama startup aplikasi dan disimpan di app.state.rag_service.")
    except Exception as e:
        print(f"ERROR KRITIS (startup): Gagal menginisialisasi RAGService: {e}")
        app.state.rag_service = None  # Pastikan tetap None jika gagal
        # Melempar error di sini akan menghentikan startup aplikasi jika RAGService gagal
        raise RuntimeError(f"Gagal menginisialisasi RAGService selama startup: {e}") from e

    yield  # Aplikasi berjalan di sini

    # Kode yang dijalankan setelah aplikasi selesai menerima request (shutdown)
    print("Memulai event lifespan: shutdown aplikasi...")
    if hasattr(app.state, 'rag_service'):
        app.state.rag_service = None  # Hapus referensi
    print("RAGService di-cleanup (referensi dihapus dari app.state).")


app = FastAPI(
    title="RAG Application with FastAPI (Azure Embeddings & OpenRouter LLM)",
    description="API untuk melakukan Retrieval-Augmented Generation menggunakan Azure OpenAI Embeddings dan LLM dari OpenRouter.",
    version="0.2.4",  # Versi update
    lifespan=lifespan
)

# Menggunakan router dari api_routes (yang merupakan modul routes.py)
app.include_router(api_routes.router, prefix="/api/v1/rag", tags=["RAG Queries"])


@app.get("/", summary="Root Endpoint", description="Endpoint selamat datang untuk RAG API.")
async def read_root():
    """
    Menampilkan pesan selamat datang.
    """
    return {"message": "Welcome to the RAG API. Gunakan endpoint /api/v1/rag/query untuk mengajukan pertanyaan."}

