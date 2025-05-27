# File: app/controllers/open_ai/controller.py

from typing import Dict, Any
from app.services.services import RAGService

class OpenAIRAGController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    async def handle_query(self, question: str) -> Dict[str, Any]:
        """Menangani query dan memanggil RAGService dengan provider Azure Chat."""
        if not self.rag_service.azure_chat_llm:
            # Validasi tambahan di level controller jika diperlukan
            raise ValueError("LLM Azure OpenAI Chat tidak dikonfigurasi atau gagal inisialisasi di RAGService.")

        # Memanggil service dengan llm_provider yang sudah ditentukan
        return await self.rag_service.answer_query(question, llm_provider="azure_chat")

