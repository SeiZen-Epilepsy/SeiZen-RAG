# SeiZen-RAG (Retrieval-Augmented Generation) with FastAPI

## Description

This project is an implementation of a Retrieval-Augmented Generation (RAG) system using FastAPI as the backend API. This system allows users to ask questions and receive answers based on the content of a collection of PDF documents. The RAG process involves retrieving relevant information from a vector database of documents and then using a Large Language Model (LLM) to generate contextual answers.

This application uses Azure OpenAI Embeddings to create vector representations of documents and queries, ChromaDB as the vector database, and LLMs from OpenRouter for answer generation.

## Key Features

- **Dynamic PDF Data Ingestion**: Ability to process all PDF files present in the `data/` directory.
- **Embedding with Azure OpenAI**: Utilizes advanced embedding models from Azure OpenAI (e.g., `text-embedding-3-small`).
- **Vector Storage with ChromaDB**: Efficiently stores and retrieves document embeddings.
- **Answer Generation with OpenRouter LLM**: Flexibility to use various LLMs available through OpenRouter (e.g., `deepseek/deepseek-chat-v3-0324:free`).
- **API with FastAPI**: Provides fast and well-documented API endpoints for interaction.
- **Source Tracking**: Generated answers include references to the source documents and pages used.
- **Centralized Configuration**: Management of configurations and API keys via a `.env` file.
- **Modular Project Structure**: Code is organized into modules for ease of management and development.

## Technology Stack

- **Backend**: Python, FastAPI
- **LLM Orchestration**: LangChain
- **Embedding Model**: Azure OpenAI Embeddings
- **Vector Database**: ChromaDB
- **Large Language Model (LLM)**: OpenRouter (e.g., DeepSeek, Mistral, etc.)
- **PDF Processing**: PyPDFLoader (from LangChain)
- **Token Calculation**: TikToken
- **Environment Management**: python-dotenv
- **ASGI Server**: Uvicorn

## Project Structure

```plaintext
    rag_project/
    ├── app/                             # FastAPI Application
    │   ├── api/                         # API Modules (routes.py)
    │   ├── core/                        # Core Configuration (config.py)
    │   ├── schemas/                     # Pydantic Schemas (schemas.py)
    │   ├── services/                    # Business Logic (services.py)
    │   └── main.py                      # FastAPI entry point
    ├── data/                            # Directory for storing source PDF files
    ├── notebooks/                       # Notebooks for development trial and error
    │   └── chunk_files/                 # Folder for save chunking file development mode
    ├── scripts/                         # Utility scripts (ingest_data.py)
    ├── vector_store/                    # Persistent ChromaDB storage
    │   └── chroma_db_azure_multi/       # Vector data
    ├── .env.example                     # Example of environment variables file
    ├── requirements.txt                 # Python dependencies
    └── README.md                        # This documentation
```

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- Pip (Python package installer)
- Access to Azure OpenAI Service (to create an embedding model deployment)
- OpenRouter Account (for LLM API key)

### 2. Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone https://github.com/SeiZen-Epilepsy/SeiZen-RAG
    cd SeiZen-RAG
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Run:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configure Environment Variables

Create an `.env` file in the root directory of the project (`./`) and fill it with the following configurations. Replace placeholders with your actual values.

```env
# Azure OpenAI Embeddings Configuration
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>"
AZURE_OPENAI_API_KEY="<your-azure-openai-api-key>"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="<your-embedding-deployment-name>"
AZURE_OPENAI_API_VERSION="2024-02-01"
OPENAI_API_KEY="<your-openai-api-key>"

# OpenRouter LLM Configuration
OPENROUTER_API_KEY="<your-openrouter-api-key>"
OPENROUTER_MODEL_NAME="deepseek/deepseek-chat-v3-0324:free"
```

## Running the Application

1. Ingest PDF Data
   Before running the FastAPI server, you need to process your PDF files and index them into ChromaDB.

   1. Place your PDF files inside the ./data/ directory.
   2. Run the data ingestion script from the root project directory:

      ```bash
      python scripts/ingest_data.py
      ```

      This script will:

      - Read all PDF files from the data/ directory. - Split them into chunks.
      - Create embeddings using Azure OpenAI.
      - Store the embeddings into ChromaDB in the `vector_store/chroma_db_azure_multi/` directory.
      - If `CLEAN_VECTOR_STORE_BEFORE_INGEST` in `ingest_data.py` is set to `True` (default), the old vector store directory will be deleted before a new ingest.

2. Running the FastAPI Server
   After the data has been successfully ingested, run the FastAPI server from the root project directory:

   ```bash
   uvicorn app.main:app --reload
   ```

   or

   ```bash
   fastapi dev app/main.py
   ```

   - `app.main:app`: Points to the `FastAPI` instance (`app`) inside the `main.py` file located in the `app` directory.

   - `--reload`: The server will automatically restart upon code changes (useful during development).

   The server will run on `http://127.0.0.1:8000` by default.

## API Endpoints

Once the server is running, you can access the interactive API documentation (Swagger UI) at:

- Swagger UI: http://127.0.0.1:8000/docs

- ReDoc: http://127.0.0.1:8000/redoc

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

MIT License is a permissive license that allows for reuse with minimal restrictions. It permits commercial use, modification, distribution, and private use while providing only the conditions that the license and copyright notice must be included with the software.
