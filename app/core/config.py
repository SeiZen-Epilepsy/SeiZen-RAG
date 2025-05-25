# File: app/core/config.py

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Tentukan path ke .env relatif terhadap file config.py ini
# config.py ada di app/core/, .env ada di root (dua level di atas)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Config: .env dimuat dari: {dotenv_path}")
else:
    print("Config: .env tidak ditemukan di path yang diharapkan, mengandalkan environment variables sistem.")


class Settings(BaseSettings):
    # Azure OpenAI Embeddings Config
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")

    # OpenRouter LLM Config
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL_NAME: str = os.getenv("OPENROUTER_MODEL_NAME")

    PROJECT_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHROMA_DB_DIR: str = os.path.join(PROJECT_ROOT_DIR, "vector_store", "chroma_db_azure_multi")
    CHROMA_COLLECTION_NAME: str = "rag_azure_multi_pdf_collection"

    RETRIEVER_SEARCH_K: int = int(os.getenv("RETRIEVER_SEARCH_K", 4))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.3))


settings = Settings()

# Validasi sederhana saat modul config dimuat (opsional, bisa juga di __init__ RAGService)
if not all([settings.AZURE_OPENAI_ENDPOINT, settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
    print("PERINGATAN (config.py): Konfigurasi Azure OpenAI Embeddings tidak lengkap.")
if not settings.OPENROUTER_API_KEY:
    print("PERINGATAN (config.py): OPENROUTER_API_KEY tidak ditemukan.")
# Pengecekan CHROMA_DB_DIR lebih baik dilakukan saat RAGService diinisialisasi.
