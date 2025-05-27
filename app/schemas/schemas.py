# File: app/schemas/schemas.py

from pydantic import BaseModel, Field
from typing import List, Any

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pertanyaan pengguna untuk RAG system.")

class SourceDocument(BaseModel):
    source_file: str = Field(description="Nama file PDF sumber.")
    page: Any = Field(description="Nomor halaman (bisa berupa angka atau 'N/A').")
    content_snippet: str = Field(description="Cuplikan singkat dari konten yang relevan.")

class QueryResponse(BaseModel):
    answer: str = Field(description="Jawaban yang dihasilkan oleh RAG system.")
    sources: List[SourceDocument] = Field(description="Daftar dokumen sumber yang digunakan untuk menghasilkan jawaban.")
