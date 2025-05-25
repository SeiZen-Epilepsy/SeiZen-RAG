import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import tiktoken

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f".env dimuat dari: {dotenv_path}")
else:
    load_dotenv()
    print("Mencoba memuat .env dari direktori saat ini atau parent.")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT_DIR, "vector_store", "chroma_db_azure_multi")
COLLECTION_NAME = "rag_azure_multi_pdf_collection"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Opsi untuk membersihkan vector store lama sebelum ingest
CLEAN_VECTOR_STORE_BEFORE_INGEST = True  # Set True untuk selalu memulai dari bersih


def validate_config():
    """Memvalidasi konfigurasi yang diperlukan."""
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
        print("ERROR: Konfigurasi Azure OpenAI Embeddings tidak lengkap.")
        print(
            "Pastikan AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME ada di .env")
        return False
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Direktori data tidak ditemukan: {DATA_DIR}")
        return False
    return True


def load_and_split_pdfs():
    """Memuat semua PDF dari DATA_DIR, memecahnya menjadi chunks."""
    all_chunks = []
    pdf_file_names = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_file_names:
        print(f"Tidak ada file PDF yang ditemukan di {DATA_DIR}")
        return []

    print(f"Ditemukan {len(pdf_file_names)} file PDF di {DATA_DIR}.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    for pdf_file_name in pdf_file_names:
        pdf_file_path = os.path.join(DATA_DIR, pdf_file_name)
        try:
            print(f"Memproses: {pdf_file_name}...")
            loader = PyPDFLoader(pdf_file_path)
            documents_from_pdf = loader.load()

            for doc in documents_from_pdf:
                # Pastikan metadata source ada dan benar
                doc.metadata["source"] = pdf_file_name
                # Jika PyPDFLoader tidak mengisi 'page', Anda mungkin perlu cara lain atau mengabaikannya
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = 'N/A'  # Default jika tidak ada

            chunks_from_doc_content = text_splitter.split_documents(documents_from_pdf)
            all_chunks.extend(chunks_from_doc_content)
            print(f" -> {len(documents_from_pdf)} halaman dimuat, {len(chunks_from_doc_content)} chunks dibuat.")
        except Exception as e:
            print(f"Error saat memproses {pdf_file_name}: {e}")

    print(f"\nTotal chunks yang dihasilkan dari semua PDF: {len(all_chunks)}")
    return all_chunks


def calculate_estimated_tokens(chunks):
    """Menghitung estimasi total token untuk semua chunks."""
    if not chunks:
        return 0
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        for chunk_doc in chunks:
            total_tokens += len(tokenizer.encode(chunk_doc.page_content))
        print(f"Total Estimasi Token untuk {len(chunks)} chunks: {total_tokens} tokens")
        # Anda bisa menambahkan estimasi biaya di sini jika mau
        return total_tokens
    except Exception as e:
        print(f"Error saat menghitung token: {e}. Pastikan 'tiktoken' terinstal.")
        return -1  # Indikasi error


def ingest_to_chromadb(chunks, embeddings_model):
    """Mengindeks chunks ke ChromaDB."""
    if not chunks:
        print("Tidak ada chunks untuk diindeks.")
        return

    if CLEAN_VECTOR_STORE_BEFORE_INGEST and os.path.exists(VECTOR_STORE_DIR):
        print(f"Membersihkan direktori vector store lama: {VECTOR_STORE_DIR}")
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
            print("Direktori lama berhasil dihapus.")
        except Exception as e:
            print(f"Error saat menghapus direktori lama: {e}")

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    print(f"Membuat atau menimpa vector store di: {VECTOR_STORE_DIR} dengan koleksi: {COLLECTION_NAME}")
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR
        )
        vector_store.persist()
        print("Data berhasil diindeks dan disimpan ke ChromaDB.")
    except Exception as e:
        print(f"Error saat mengindeks data ke ChromaDB: {e}")


def main():
    print("Memulai proses ingest data...")
    if not validate_config():
        return

    # 1. Muat dan pecah PDF menjadi chunks
    all_pdf_chunks = load_and_split_pdfs()
    if not all_pdf_chunks:
        print("Proses ingest dihentikan karena tidak ada chunks yang dihasilkan.")
        return

    # 2. (Opsional) Hitung estimasi token
    calculate_estimated_tokens(all_pdf_chunks)

    # 3. Inisialisasi model embedding Azure
    try:
        azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )
        print(f"Model Azure OpenAI Embeddings ('{AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}') berhasil diinisialisasi.")
    except Exception as e:
        print(f"Error saat menginisialisasi AzureOpenAIEmbeddings: {e}")
        return

    # 4. Ingest chunks ke ChromaDB
    ingest_to_chromadb(all_pdf_chunks, azure_embeddings)

    print("Proses ingest data selesai.")


if __name__ == "__main__":
    main()
