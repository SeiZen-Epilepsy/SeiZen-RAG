# File: app/api/routes/routes.py

from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Annotated

# Menggunakan path impor yang sesuai dengan struktur pengguna
try:
    from ...schemas.schemas import QueryRequest, QueryResponse  # Dari app/schemas/schemas.py
    from ...services.services import RAGService  # Dari app/services/services.py
except ImportError:
    # Fallback jika dijalankan dengan cara yang berbeda atau struktur sedikit berbeda
    # Ini mungkin tidak diperlukan jika struktur proyek konsisten
    from app.schemas.schemas import QueryRequest, QueryResponse
    from app.services.services import RAGService

router = APIRouter()


# Fungsi dependency untuk mengambil RAGService dari app.state
def get_rag_service_from_app_state(request: Request) -> RAGService:
    if not hasattr(request.app.state, 'rag_service') or request.app.state.rag_service is None:
        print(
            "ERROR (get_rag_service_from_app_state di routes.py): RAGService tidak ditemukan di app.state atau belum diinisialisasi.")
        # Ini akan terjadi jika lifespan gagal menginisialisasi RAGService
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Layanan RAG tidak tersedia saat ini (gagal inisialisasi atau startup)."
        )
    return request.app.state.rag_service


# Menggunakan Annotated untuk dependency injection yang lebih modern di FastAPI
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service_from_app_state)]


@router.post("/query",
             response_model=QueryResponse,
             summary="Ajukan Pertanyaan ke Sistem RAG",
             description="Mengirimkan pertanyaan ke sistem RAG dan menerima jawaban beserta sumber yang digunakan.")
async def ask_rag_question(
        request_data: QueryRequest,
        service: RAGServiceDep  # Menggunakan dependency injection dari app.state
):
    """
    Endpoint untuk mengajukan pertanyaan ke sistem RAG.
    - **question**: Pertanyaan yang ingin Anda ajukan.
    """
    if not request_data.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pertanyaan tidak boleh kosong."
        )

    try:
        print(f"Menerima query: {request_data.question}")
        result = await service.answer_query(request_data.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except FileNotFoundError as e:
        print(f"Error spesifik (FileNotFoundError) saat memproses query: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Layanan RAG tidak siap. Database sumber mungkin belum diinisialisasi. Detail: {str(e)}"
        )
    except RuntimeError as e:
        print(f"Error spesifik (RuntimeError) saat memproses query: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Layanan RAG mengalami masalah internal. Detail: {str(e)}"
        )
    except Exception as e:
        print(f"Error tidak terduga saat memproses query RAG: {e}")
        # import traceback
        # traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan internal saat memproses pertanyaan Anda."
        )
