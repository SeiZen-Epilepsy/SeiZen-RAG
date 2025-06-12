import os
import time
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from langchain_chroma import Chroma  # Updated import
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, Dict, Any, Literal, Optional
from app.core.config import settings
from app.core.logging_config import logger
from app.core.exceptions import (
    ConfigurationError, EmbeddingModelError, VectorStoreError,
    LLMProviderError, RetrieverError, QueryProcessingError
)

LLMProviderType = Literal["azure_chat", "openrouter"]

class RAGService:
    def __init__(self):
        logger.info("Initializing RAGService...")
        self._initialize_components()
        logger.info("RAGService fully initialized successfully")

    def _initialize_components(self):
        """Initialize all RAGService components with proper error handling."""
        self._validate_configuration()
        self._initialize_embeddings()
        self._initialize_vector_store()
        self._initialize_llm_clients()
        self._initialize_retriever()

    def _validate_configuration(self):
        """Validate all required configurations."""
        if not all([settings.AZURE_OPENAI_EMBEDDING_ENDPOINT, settings.AZURE_OPENAI_EMBEDDING_API_KEY,
                    settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
            raise ConfigurationError("Azure OpenAI Embeddings configuration is incomplete")

        chroma_db_file_path = os.path.join(settings.CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.isdir(settings.CHROMA_DB_DIR) or not os.path.exists(chroma_db_file_path):
            raise ConfigurationError(
                f"ChromaDB database not found at: {settings.CHROMA_DB_DIR}. Run 'scripts/ingest_data.py'")

    def _initialize_embeddings(self):
        """Initialize embedding model with error handling."""
        try:
            self.embeddings_model = AzureOpenAIEmbeddings(
                azure_endpoint=settings.AZURE_OPENAI_EMBEDDING_ENDPOINT,
                openai_api_key=settings.AZURE_OPENAI_EMBEDDING_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                api_version=settings.AZURE_OPENAI_EMBEDDING_API_VERSION,
            )
            logger.info("Azure OpenAI Embeddings model successfully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Embeddings model: {e}")
            raise EmbeddingModelError(f"Failed to initialize Azure Embeddings model: {e}") from e

    def _initialize_vector_store(self):
        """Initialize vector store with error handling."""
        try:
            self.vector_store = Chroma(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings_model,
                persist_directory=settings.CHROMA_DB_DIR
            )
            logger.info(f"ChromaDB vector store successfully loaded from: {settings.CHROMA_DB_DIR}")
        except Exception as e:
            logger.error(f"Failed to load ChromaDB vector store: {e}")
            raise VectorStoreError(f"Failed to load ChromaDB vector store: {e}") from e

    def _initialize_llm_clients(self):
        """Initialize LLM clients with error handling."""
        self.azure_chat_llm: Optional[AzureChatOpenAI] = None
        self.openrouter_llm: Optional[ChatOpenAI] = None
        
        # Initialize Azure Chat LLM
        if all([settings.AZURE_OPENAI_CHAT_ENDPOINT, settings.AZURE_OPENAI_CHAT_API_KEY,
                settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
            try:
                self.azure_chat_llm = AzureChatOpenAI(
                    azure_endpoint=settings.AZURE_OPENAI_CHAT_ENDPOINT,
                    openai_api_key=settings.AZURE_OPENAI_CHAT_API_KEY,
                    azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                    model_name=settings.AZURE_OPENAI_CHAT_MODEL_NAME,
                    api_version=settings.AZURE_OPENAI_CHAT_API_VERSION,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS if settings.LLM_MAX_TOKENS else None
                )
                logger.info(f"Azure OpenAI Chat LLM ({settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}) successfully initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure OpenAI Chat LLM: {e}")
        else:
            logger.info("Azure OpenAI Chat LLM configuration is incomplete")

        # Initialize OpenRouter LLM
        if settings.OPENROUTER_API_KEY:
            try:
                self.openrouter_llm = ChatOpenAI(
                    openai_api_key=settings.OPENROUTER_API_KEY,
                    base_url=settings.OPENROUTER_ENDPOINT,
                    model_name=settings.OPENROUTER_MODEL_NAME,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS if settings.LLM_MAX_TOKENS else None
                )
                logger.info(f"OpenRouter LLM ({settings.OPENROUTER_MODEL_NAME}) successfully initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter LLM: {e}")
        else:
            logger.info("OPENROUTER_API_KEY is not configured")

        if not self.azure_chat_llm and not self.openrouter_llm:
            raise LLMProviderError("No LLM (Azure Chat or OpenRouter) was successfully configured")

    def _initialize_retriever(self):
        """Initialize retriever with error handling."""
        try:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.RETRIEVER_SEARCH_K}
            )
            logger.info("Retriever successfully created")
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise RetrieverError(f"Failed to create retriever: {e}") from e

    def _format_docs_for_context(self, docs: List[Document]) -> str:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Source: {source}, Page: {str(page)}]\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)

    def _build_rag_chain(self, llm_client: Any):
        if llm_client is None:
            raise ValueError("LLM client not provided or not initialized for _build_rag_chain.")

        prompt_template_str = """
        You are a very helpful AI assistant. Use the following context snippets to answer the user's question.
        The context comes from various documents, sources and pages will be listed.
        Answer the question based only on the provided context.
        If the information is not available in the context, say you cannot find the answer in the provided documents.
        Answer clearly and concisely.

        Context:
        {context}

        Question:
        {question}

        Answer (based on the context above):
        """
        prompt = PromptTemplate.from_template(prompt_template_str)

        return (
                RunnableParallel(
                    {"context_docs": self.retriever, "question": RunnablePassthrough()}
                )
                | RunnableParallel(
            {
                "answer": (
                        RunnablePassthrough()
                        | RunnablePassthrough.assign(context=lambda x: self._format_docs_for_context(x["context_docs"]))
                        | prompt
                        | llm_client
                        | StrOutputParser()
                ),
                "sources": lambda x: x["context_docs"]
            }
        )
        )

    async def answer_query(self, question: str, llm_provider: LLMProviderType) -> Dict[str, Any]:
        """Process query with enhanced error handling and timing."""
        start_time = time.time()
        
        try:
            chosen_llm = self._get_llm_client(llm_provider)
            rag_chain = self._build_rag_chain(chosen_llm)
            
            logger.info(f"Processing query with {llm_provider}: {question[:100]}...")
            result = await rag_chain.ainvoke(question)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            formatted_sources = self._format_sources(result.get("sources", []))
            
            response = {
                "answer": result.get("answer", "No answer generated."), 
                "sources": formatted_sources,
                "query_metadata": {
                    "llm_provider": llm_provider,
                    "retriever_k": settings.RETRIEVER_SEARCH_K,
                    "processing_time_ms": processing_time
                }
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}ms using {llm_provider}")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error processing query with {llm_provider} (took {processing_time:.2f}ms): {e}")
            raise QueryProcessingError(f"Failed to process query with {llm_provider} LLM") from e

    def _get_llm_client(self, llm_provider: LLMProviderType) -> Any:
        """Get LLM client with validation."""
        if llm_provider == "azure_chat":
            if not self.azure_chat_llm:
                raise LLMProviderError("Azure OpenAI Chat LLM is not configured or failed initialization")
            return self.azure_chat_llm
        elif llm_provider == "openrouter":
            if not self.openrouter_llm:
                raise LLMProviderError("OpenRouter LLM is not configured or failed initialization")
            return self.openrouter_llm
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

    def _format_sources(self, sources_data: List) -> List[Dict[str, Any]]:
        """Format source documents with enhanced metadata."""
        formatted_sources = []
        if isinstance(sources_data, list):
            for doc in sources_data:
                if isinstance(doc, Document) and hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    formatted_sources.append({
                        "source_file": doc.metadata.get('source', 'N/A'),
                        "page": str(doc.metadata.get('page', 'N/A')),
                        "content_snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
                        "relevance_score": doc.metadata.get('score')  # If available
                    })
        return formatted_sources

    def health_check(self) -> Dict[str, str]:
        """Perform health check on all components."""
        health_status = {}
        
        # Check embeddings model
        try:
            if self.embeddings_model:
                health_status["embeddings"] = "healthy"
            else:
                health_status["embeddings"] = "not_initialized"
        except Exception:
            health_status["embeddings"] = "unhealthy"

        # Check vector store
        try:
            if self.vector_store:
                health_status["vector_store"] = "healthy"
            else:
                health_status["vector_store"] = "not_initialized"
        except Exception:
            health_status["vector_store"] = "unhealthy"

        # Check LLM providers
        health_status["azure_chat_llm"] = "healthy" if self.azure_chat_llm else "not_configured"
        health_status["openrouter_llm"] = "healthy" if self.openrouter_llm else "not_configured"

        return health_status

