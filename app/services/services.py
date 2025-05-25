# File: app/services/services.py

import os
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, Dict, Any

# Mengimpor settings terpusat
try:
    from ..core.config import settings
except ImportError:  # Fallback untuk testing atau struktur berbeda
    from core.config import settings


class RAGService:
    def __init__(self):
        # Konstruktor sekarang akan dijalankan sekali oleh lifespan di main.py
        print("Menginisialisasi RAGService...")

        # 1. Validasi Konfigurasi Awal
        if not all([settings.AZURE_OPENAI_ENDPOINT, settings.AZURE_OPENAI_API_KEY,
                    settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
            raise ValueError(
                "Konfigurasi Azure OpenAI Embeddings tidak lengkap. Periksa file .env dan app.core.config.")
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY tidak ditemukan. Periksa file .env dan app.core.config.")

        chroma_db_file_path = os.path.join(settings.CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.isdir(settings.CHROMA_DB_DIR) or not os.path.exists(chroma_db_file_path):
            raise FileNotFoundError(
                f"Direktori atau file database ChromaDB ('chroma.sqlite3') tidak ditemukan di: {settings.CHROMA_DB_DIR}. "
                "Pastikan skrip 'scripts/ingest_data.py' sudah dijalankan dan berhasil membuat database."
            )

        # 2. Inisialisasi Model Embedding (Azure)
        try:
            self.embeddings_model = AzureOpenAIEmbeddings(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            )
            print("Model Azure OpenAI Embeddings berhasil diinisialisasi untuk RAG service.")
        except Exception as e:
            print(f"Error kritis saat menginisialisasi Azure Embeddings: {e}")
            raise RuntimeError("Gagal menginisialisasi Azure Embeddings model.") from e

        # 3. Muat Vector Store (ChromaDB)
        try:
            self.vector_store = Chroma(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings_model,
                persist_directory=settings.CHROMA_DB_DIR
            )
            print(f"ChromaDB vector store berhasil dimuat dari: {settings.CHROMA_DB_DIR}")
        except Exception as e:
            print(f"Error kritis saat memuat ChromaDB: {e}")
            raise RuntimeError("Gagal memuat ChromaDB vector store.") from e

        # 4. Inisialisasi LLM (OpenRouter)
        try:
            self.llm = ChatOpenAI(
                openai_api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                model_name=settings.OPENROUTER_MODEL_NAME,
                temperature=settings.LLM_TEMPERATURE,
            )
            print(f"LLM OpenRouter ({settings.OPENROUTER_MODEL_NAME}) berhasil diinisialisasi.")
        except Exception as e:
            print(f"Error kritis saat menginisialisasi OpenRouter LLM: {e}")
            raise RuntimeError("Gagal menginisialisasi OpenRouter LLM.") from e

        # 5. Membuat Retriever
        try:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.RETRIEVER_SEARCH_K}
            )
            print("Retriever berhasil dibuat.")
        except Exception as e:
            print(f"Error kritis saat membuat retriever: {e}")
            raise RuntimeError("Gagal membuat retriever dari vector store.") from e

        # 6. Membangun RAG Chain (LCEL)
        self._build_rag_chain()
        print("RAG chain berhasil dibangun.")
        print("RAGService berhasil diinisialisasi sepenuhnya.")

    def _format_docs_for_context(self, docs: List[Document]) -> str:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Tidak Diketahui')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Sumber: {source}, Halaman: {str(page)}]\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)

    def _build_rag_chain(self):
        prompt_template_str = """
        You are a very helpful AI assistant. Use the following context to answer user questions. The context comes from various documents. Answer the questions solely based on the provided context. If the information is not available in the context, say you cannot find the answer in the provided documents. Try to keep your answers concise and to the point. If possible and relevant, mention from which source document the information comes (use metadata 'source').
    
        Context:{context}
    
        Question:{question}
    
        Answer (based on the above context):
        """
        prompt = PromptTemplate.from_template(prompt_template_str)

        self.rag_chain = (
                RunnableParallel(
                    {"context_docs": self.retriever, "question": RunnablePassthrough()}
                )
                | RunnableParallel(
            {
                "answer": (
                        RunnablePassthrough()
                        | RunnablePassthrough.assign(context=lambda x: self._format_docs_for_context(x["context_docs"]))
                        | prompt
                        | self.llm
                        | StrOutputParser()
                ),
                "sources": lambda x: x["context_docs"]
            }
        )
        )

    async def answer_query(self, question: str) -> Dict[str, Any]:
        if not self.rag_chain:
            print("ERROR: RAG chain belum diinisialisasi saat answer_query dipanggil.")
            raise RuntimeError("RAG chain tidak siap. Inisialisasi mungkin gagal.")

        try:
            result = await self.rag_chain.ainvoke(question)

            formatted_sources = []
            sources_data = result.get("sources")
            if isinstance(sources_data, list):
                for doc in sources_data:
                    if isinstance(doc, Document) and hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                        formatted_sources.append({
                            "source_file": doc.metadata.get('source', 'N/A'),
                            "page": str(doc.metadata.get('page', 'N/A')),
                            "content_snippet": doc.page_content[:250] + "..."
                        })
                    else:
                        print(f"Peringatan: Item tidak valid dalam sources_data: {type(doc)}")

            return {"answer": result.get("answer", "Tidak ada jawaban yang dihasilkan."), "sources": formatted_sources}
        except Exception as e:
            print(f"Error dalam RAG chain invocation: {e}")
            raise
