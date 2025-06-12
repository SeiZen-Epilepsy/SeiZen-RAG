from typing import Dict, Any
from app.services.services import RAGService

class OpenAIRAGController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    async def handle_query(self, question: str) -> Dict[str, Any]:
        """Handle query and call RAGService with Azure Chat provider."""
        if not self.rag_service.azure_chat_llm:
            raise ValueError("Azure OpenAI Chat LLM is not configured or failed initialization in RAGService.")

        # Call service with predetermined llm_provider
        return await self.rag_service.answer_query(question, llm_provider="azure_chat")

