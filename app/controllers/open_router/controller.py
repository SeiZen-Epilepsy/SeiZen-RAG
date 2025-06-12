from typing import Dict, Any
from app.services.services import RAGService

class OpenRouterRAGController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    async def handle_query(self, question: str) -> Dict[str, Any]:
        """Handle query and call RAGService with OpenRouter provider."""
        if not self.rag_service.openrouter_llm:
            raise ValueError("OpenRouter LLM is not configured or failed initialization in RAGService.")

        return await self.rag_service.answer_query(question, llm_provider="openrouter")

