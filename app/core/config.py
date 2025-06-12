import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional, Any

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Config: .env loaded from: {dotenv_path}")
else:
    print("Config: .env not found, relying on system environment variables.")

class Settings(BaseSettings):
    # Azure OpenAI Embeddings Config
    AZURE_OPENAI_EMBEDDING_ENDPOINT: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
    AZURE_OPENAI_EMBEDDING_API_KEY: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01")

    # Azure OpenAI Chat LLM Config
    AZURE_OPENAI_CHAT_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "") # From your .env, this is for chat
    AZURE_OPENAI_CHAT_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")    # From your .env, this is for chat
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "")
    AZURE_OPENAI_CHAT_MODEL_NAME: str = os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4o-mini")
    AZURE_OPENAI_CHAT_API_VERSION: str = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2024-02-15-preview")  # More stable version

    # OpenRouter LLM Config
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL_NAME: str = os.getenv("OPENROUTER_MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")
    OPENROUTER_ENDPOINT: str = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1") # Add this

    PROJECT_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHROMA_DB_DIR: str = os.path.join(PROJECT_ROOT_DIR, "vector_store", "chroma_db_azure_multi")
    CHROMA_COLLECTION_NAME: str = "rag_azure_multi_pdf_collection"

    RETRIEVER_SEARCH_K: int = int(os.getenv("RETRIEVER_SEARCH_K", 4))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.3))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 800))

settings = Settings()

# Validation
if not all([settings.AZURE_OPENAI_EMBEDDING_ENDPOINT, settings.AZURE_OPENAI_EMBEDDING_API_KEY, settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
    print("WARNING (config.py): Azure OpenAI Embeddings configuration is incomplete.")
if not all([settings.AZURE_OPENAI_CHAT_ENDPOINT, settings.AZURE_OPENAI_CHAT_API_KEY, settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
    print("WARNING (config.py): Azure OpenAI Chat LLM configuration is incomplete.")
if not settings.OPENROUTER_API_KEY:
    print("WARNING (config.py): OPENROUTER_API_KEY not found.")
