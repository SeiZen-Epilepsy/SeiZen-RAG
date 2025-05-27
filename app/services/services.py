# File: app/services/services.py

import os
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, Dict, Any, Literal, Optional
from app.core.config import settings

LLMProviderType = Literal["azure_chat", "openrouter"]  # Hanya Azure Chat dan OpenRouter

class RAGService:
    def __init__(self):
        print("Menginisialisasi RAGService...")

        # --- Validasi Konfigurasi ---
        if not all([settings.AZURE_OPENAI_EMBEDDING_ENDPOINT, settings.AZURE_OPENAI_EMBEDDING_API_KEY,
                    settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
            raise ValueError("Konfigurasi Azure OpenAI Embeddings tidak lengkap.")

        chroma_db_file_path = os.path.join(settings.CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.isdir(settings.CHROMA_DB_DIR) or not os.path.exists(chroma_db_file_path):
            raise FileNotFoundError(
                f"Database ChromaDB tidak ditemukan di: {settings.CHROMA_DB_DIR}. Jalankan 'scripts/ingest_data.py'.")

        # --- Inisialisasi Embedding Model (Azure) ---
        try:
            self.embeddings_model = AzureOpenAIEmbeddings(
                azure_endpoint=settings.AZURE_OPENAI_EMBEDDING_ENDPOINT,
                openai_api_key=settings.AZURE_OPENAI_EMBEDDING_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                api_version=settings.AZURE_OPENAI_EMBEDDING_API_VERSION,
            )
            print("Model Azure OpenAI Embeddings berhasil diinisialisasi.")
        except Exception as e:
            raise RuntimeError(f"Gagal menginisialisasi Azure Embeddings model: {e}") from e

        # --- Muat Vector Store (ChromaDB) ---
        try:
            self.vector_store = Chroma(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings_model,
                persist_directory=settings.CHROMA_DB_DIR
            )
            print(f"ChromaDB vector store berhasil dimuat dari: {settings.CHROMA_DB_DIR}")
        except Exception as e:
            raise RuntimeError(f"Gagal memuat ChromaDB vector store: {e}") from e

        # --- Inisialisasi Klien LLM ---
        self.azure_chat_llm: Optional[AzureChatOpenAI] = None
        if all([settings.AZURE_OPENAI_CHAT_ENDPOINT, settings.AZURE_OPENAI_CHAT_API_KEY,
                settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
            try:
                model_kwargs_azure = {}
                if settings.LLM_MAX_TOKENS: model_kwargs_azure["max_tokens"] = settings.LLM_MAX_TOKENS
                self.azure_chat_llm = AzureChatOpenAI(
                    azure_endpoint=settings.AZURE_OPENAI_CHAT_ENDPOINT,
                    openai_api_key=settings.AZURE_OPENAI_CHAT_API_KEY,
                    azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                    model_name=settings.AZURE_OPENAI_CHAT_MODEL_NAME,
                    api_version=settings.AZURE_OPENAI_CHAT_API_VERSION,
                    temperature=settings.LLM_TEMPERATURE,
                    model_kwargs=model_kwargs_azure if model_kwargs_azure else None
                )
                print(f"LLM Azure OpenAI Chat ({settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}) berhasil diinisialisasi.")
            except Exception as e:
                print(f"PERINGATAN: Gagal menginisialisasi LLM Azure OpenAI Chat: {e}")
        else:
            print("INFO: Konfigurasi Azure OpenAI Chat LLM tidak lengkap.")

        self.openrouter_llm: Optional[ChatOpenAI] = None
        if settings.OPENROUTER_API_KEY:
            try:
                model_kwargs_openrouter = {}
                if settings.LLM_MAX_TOKENS: model_kwargs_openrouter["max_tokens"] = settings.LLM_MAX_TOKENS
                self.openrouter_llm = ChatOpenAI(
                    openai_api_key=settings.OPENROUTER_API_KEY,
                    base_url=settings.OPENROUTER_ENDPOINT,  # Menggunakan settings.OPENROUTER_ENDPOINT
                    model_name=settings.OPENROUTER_MODEL_NAME,
                    temperature=settings.LLM_TEMPERATURE,
                    model_kwargs=model_kwargs_openrouter
                )
                print(f"LLM OpenRouter ({settings.OPENROUTER_MODEL_NAME}) berhasil diinisialisasi.")
            except Exception as e:
                print(f"PERINGATAN: Gagal menginisialisasi LLM OpenRouter: {e}")
        else:
            print("INFO: OPENROUTER_API_KEY tidak dikonfigurasi.")

        if not self.azure_chat_llm and not self.openrouter_llm:
            raise RuntimeError(
                "Tidak ada LLM (Azure Chat atau OpenRouter) yang berhasil dikonfigurasi. RAG service tidak dapat berfungsi.")

        # --- Membuat Retriever ---
        try:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.RETRIEVER_SEARCH_K}
            )
            print("Retriever berhasil dibuat.")
        except Exception as e:
            raise RuntimeError(f"Gagal membuat retriever: {e}") from e

        print("RAGService berhasil diinisialisasi sepenuhnya.")

    def _format_docs_for_context(self, docs: List[Document]) -> str:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Tidak Diketahui')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Sumber: {source}, Halaman: {str(page)}]\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)

    def _build_rag_chain(self, llm_client: Any):
        if llm_client is None:
            raise ValueError("Klien LLM tidak disediakan atau tidak terinisialisasi untuk _build_rag_chain.")

        prompt_template_str = """
        Anda adalah asisten AI yang sangat membantu. Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna.
        Konteks berasal dari berbagai dokumen, sumber dan halaman akan dicantumkan.
        Jawablah pertanyaan hanya berdasarkan konteks yang diberikan.
        Jika informasi tidak ada dalam konteks, katakan Anda tidak dapat menemukan jawabannya dalam dokumen yang disediakan.
        Jawablah dengan jelas dan ringkas.

        Konteks:
        {context}

        Pertanyaan:
        {question}

        Jawaban (berdasarkan konteks di atas):
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
        chosen_llm: Optional[Any] = None
        if llm_provider == "azure_chat":
            chosen_llm = self.azure_chat_llm
            if not chosen_llm:
                raise ValueError("LLM Azure OpenAI Chat tidak dikonfigurasi atau gagal inisialisasi.")
        elif llm_provider == "openrouter":
            chosen_llm = self.openrouter_llm
            if not chosen_llm:
                raise ValueError("LLM OpenRouter tidak dikonfigurasi atau gagal inisialisasi.")
        else:
            # Ini seharusnya tidak dipanggil jika controller hanya mengirim tipe yang valid
            raise ValueError(f"Penyedia LLM tidak dikenal: {llm_provider}. Pilih 'azure_chat' atau 'openrouter'.")

        rag_chain = self._build_rag_chain(chosen_llm)

        try:
            result = await rag_chain.ainvoke(question)

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

            return {"answer": result.get("answer", "Tidak ada jawaban yang dihasilkan."), "sources": formatted_sources}
        except Exception as e:
            print(f"Error dalam RAG chain invocation dengan {llm_provider}: {e}")
            raise RuntimeError(f"Gagal memproses query dengan {llm_provider} LLM.") from e

